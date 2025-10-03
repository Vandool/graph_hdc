import math
from collections.abc import Callable

import networkx as nx
from rdkit import Chem
from rdkit.Chem import QED, AllChem, Crippen, DataStructs

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.utils.chem import canonical_key, is_valid_molecule, reconstruct_for_eval
from src.utils.utils import pick_device


def rdkit_logp(m: Chem.Mol) -> float:
    return Crippen.MolLogP(m)


def rdkit_qed(m: Chem.Mol) -> float:
    return QED.qed(m)


class GenerationEvaluator:
    def __init__(self, base_dataset: str, device=None):
        self.device = device if device is not None else pick_device()
        self.base_dataset = base_dataset

        dataset = ZincSmiles(split="train") if base_dataset == "zinc" else QM9Smiles(split="train")
        self.T = {d.eval_smiles for d in dataset}.union({d.smiles for d in dataset})

        self.mols: list[Chem.Mol | None] | None = None
        self.valid_flags: list[bool] | None = None

    def _to_mols_and_valid(self, samples: list[nx.Graph]) -> tuple[list[Chem.Mol | None], list[bool]]:
        mols: list[Chem.Mol | None] = []
        for g in samples:
            try:
                mols.append(reconstruct_for_eval(g, dataset=self.base_dataset))
            except Exception as e:
                print(f"nx_to_mol error: {e}")
                mols.append(None)
        valid_flags = [(m is not None and is_valid_molecule(m)) for m in mols]
        return mols, valid_flags

    def evaluate(self, samples: list[nx.Graph], final_flags: list[bool], sims: list[list[float]]) -> dict[str, float]:
        n_samples = len(samples)

        def sim_stats(values: list[float], prefix: str) -> dict[str, float]:
            if not values:
                return {f"{prefix}_sim_mean": 0.0, f"{prefix}_sim_min": 0.0, f"{prefix}_sim_max": 0.0}
            return {
                f"{prefix}_sim_mean": sum(values) / len(values),
                f"{prefix}_sim_min": min(values),
                f"{prefix}_sim_max": max(values),
            }

        # split sims by final vs nonfinal
        final_sims, non_final_sims = [], []
        for flag, s in zip(final_flags, sims, strict=False):
            best = max(s) if s else 0.0
            (final_sims if flag else non_final_sims).append(best)

        sims_eval = {}
        sims_eval.update(sim_stats(final_sims, "final"))
        sims_eval.update(sim_stats(non_final_sims, "nonfinal"))

        # mols + validity (and store on self)
        mols, valid_flags = self._to_mols_and_valid(samples)
        self.mols = mols
        self.valid_flags = valid_flags

        n_valid = sum(valid_flags)
        validity = 100.0 * n_valid / n_samples if n_samples else 0.0

        # uniqueness / novelty
        valid_canon = [canonical_key(m) for m, f in zip(mols, valid_flags, strict=False) if f]
        valid_canon = [c for c in valid_canon if c is not None]
        unique_valid = set(valid_canon)
        uniqueness = 100.0 * len(unique_valid) / n_valid if n_valid else 0.0

        novel_set = unique_valid - self.T
        novelty = 100.0 * len(novel_set) / n_valid if n_valid else 0.0
        nuv = 100.0 * len(novel_set) / n_samples if n_samples else 0.0

        return {
            "dataset": self.base_dataset,
            "final_flags": 100.0 * sum(final_flags) / n_samples if n_samples else 0.0,
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty": novelty,
            "nuv": nuv,
            **sims_eval,
        }

    def evaluate_conditional(
        self,
        samples,  # list[nx.Graph]
        target: float,  # single float target
        final_flags: list[bool],
        prop_fn: Callable = rdkit_logp,
        eps: float = 0.2,
        compute_diversity: bool = True,
        total_samples: int = 100,
    ) -> dict[str, dict[str, float]]:
        """
        Returns a dict with stable sections:
          - meta:   dataset and configuration
          - total:  metrics normalized by total_samples
          - valid:  metrics over valid, non-NaN property samples
          - hits:   metrics over the hit subset (|prop-target| <= eps among valid)
        """

        # ---------- meta ----------
        out = {
            "meta": {
                "dataset": self.base_dataset,
                "n_samples": len(samples),
                "total_samples": int(total_samples),
                "target": float(target),
                "epsilon": float(eps),
            },
            "total": {
                "validity_pct": 0.0,
                "final_pct": 100.0 * sum(final_flags) / total_samples if total_samples else 0.0,
            },
            "valid": {
                "n_valid": 0,
                "n_valid_non_nan": 0,
                "mae_to_target": float("nan"),
                "rmse_to_target": float("nan"),
                "success_at_eps_pct": 0.0,  # den = n_valid_non_nan
                "final_success_at_eps_pct": 0.0,  # den = n_valid_non_nan
                "uniqueness_pct": 0.0,  # among valid
                "novelty_pct": 0.0,  # among valid
            },
            "hits": {
                "n_hits": 0,
                "uniqueness_hits_pct": 0.0,  # among hits
                "novelty_hits_pct": 0.0,  # among hits
                "diversity_hits_pct": 0.0,  # 100 * (1 - mean tanimoto sim) over hits
            },
        }

        # ---------- to mols & validity ----------
        mols, valid = self._to_mols_and_valid(samples)
        self.mols = mols
        self.valid_flags = valid

        n_valid = int(sum(valid))
        out["valid"]["n_valid"] = n_valid
        out["total"]["validity_pct"] = 100.0 * n_valid / total_samples if total_samples else 0.0

        if n_valid == 0:
            return out

        # ---------- compute property on valid only; keep mapping to valid indices ----------
        props_triplets = []  # (valid_idx, prop, final_flag)
        tgt = float(target)
        v_idx = -1
        for i, (m, v, f) in enumerate(zip(mols, valid, final_flags, strict=False)):
            if not v:
                continue
            v_idx += 1  # position among *valid* items
            try:
                p = float(prop_fn(m))
            except Exception:
                p = float("nan")
            props_triplets.append((v_idx, p, bool(f)))

        # filter non-NaN props, keep mapping to valid index
        paired = [(vidx, p, tgt, f) for (vidx, p, f) in props_triplets if not math.isnan(p)]
        den = len(paired)
        out["valid"]["n_valid_non_nan"] = den
        if den == 0:
            return out

        # ---------- absolute errors & success ----------
        abs_err = [abs(p - t) for (_, p, t, _) in paired]
        finals = [f for (_, _, _, f) in paired]

        out["valid"]["mae_to_target"] = sum(abs_err) / den
        out["valid"]["rmse_to_target"] = math.sqrt(sum(e * e for e in abs_err) / den)
        out["valid"]["success_at_eps_pct"] = 100.0 * sum(e <= eps for e in abs_err) / den
        out["valid"]["final_success_at_eps_pct"] = (
            100.0 * sum((e <= eps) and f for e, f in zip(abs_err, finals, strict=False)) / den
        )

        # ---------- uniqueness / novelty over *valid* ----------
        valid_canon = [canonical_key(m) for m, v in zip(mols, valid, strict=False) if v]
        valid_canon = [c for c in valid_canon if c is not None]
        if valid_canon:
            uniq = 100.0 * len(set(valid_canon)) / len(valid_canon)
            novel = 100.0 * len(set(valid_canon) - self.T) / len(valid_canon)
            out["valid"]["uniqueness_pct"] = uniq
            out["valid"]["novelty_pct"] = novel

        # ---------- hit subset (|prop-target| <= eps) ----------
        hit_paired = [(vidx, p) for (vidx, p, _, _) in paired if abs(p - tgt) <= eps]
        out["hits"]["n_hits"] = len(hit_paired)
        if not hit_paired:
            return out

        # map hit valid indices back to canonical keys / mols
        hit_valid_idx = [vidx for (vidx, _) in hit_paired]
        hit_keys = []
        # valid_canon is in the same order as valid==True items; indices match vidx
        for vidx in hit_valid_idx:
            if 0 <= vidx < len(valid_canon):
                k = valid_canon[vidx]
                if k is not None:
                    hit_keys.append(k)

        if hit_keys:
            n_hits = len(hit_paired)
            out["hits"]["uniqueness_hits_pct"] = 100.0 * len(set(hit_keys)) / n_hits
            out["hits"]["novelty_hits_pct"] = 100.0 * len(set(hit_keys) - self.T) / n_hits

        if compute_diversity and len(hit_valid_idx) >= 2:
            # rebuild list of *valid* mol indices to map vidx -> original mol index
            valid_orig_idx = [i for i, v in enumerate(valid) if v]
            hit_orig_idx = [valid_orig_idx[vidx] for vidx in hit_valid_idx]
            hit_mols = [mols[i] for i in hit_orig_idx]
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in hit_mols]
            sims = [
                DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i in range(len(fps)) for j in range(i + 1, len(fps))
            ]
            if sims:
                out["hits"]["diversity_hits_pct"] = 100.0 * (1.0 - (sum(sims) / len(sims)))

        return out

    def get_mols_and_valid_flags(self):
        return self.mols, self.valid_flags
