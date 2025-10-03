import math

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
        prop_fn=rdkit_logp,
        eps: float = 0.2,
        compute_diversity: bool = True,
        total_samples: int = 100,
    ) -> dict[str, float]:
        # --- template of all outputs ---
        out = {
            "dataset": self.base_dataset,
            "n_samples": len(samples),
            "validity": 0.0,
            "final_flags": 100.0 * sum(final_flags) / total_samples,
            "target_eps": eps,
            "success@eps": 0.0,
            "final_success@eps": 0.0,
            "mae_to_target": float("nan"),
            "rmse_to_target": float("nan"),
            "corr_spearman": float("nan"),
            "uniqueness": 0.0,  # among valids
            "novelty": 0.0,  # among valids
            "uniqueness_hits": 0.0,
            "novelty_hits": 0.0,
            "diversity_hits": 0.0,
        }

        # --- mols + validity ---
        mols, valid = self._to_mols_and_valid(samples)
        self.mols = mols
        self.valid_flags = valid

        n_valid = sum(valid)
        out["validity"] = 100.0 * n_valid / total_samples

        if n_valid == 0:
            return out

        # --- compute property ---
        props, tgts, finals = [], [], []
        for m, v, f in zip(mols, valid, final_flags, strict=False):
            if v:
                try:
                    props.append(float(prop_fn(m)))
                except Exception:
                    props.append(float("nan"))
                finally:
                    tgts.append(float(target))
                    finals.append(f)

        # --- filter nans ---
        paired = [(p, t, f) for p, t, f in zip(props, tgts, finals, strict=False) if not math.isnan(p)]
        if not paired:
            return out

        fs = [f for _, _, f in paired]
        abs_err = [abs(p - t) for p, t, _ in paired]

        out["mae_to_target"] = sum(abs_err) / len(abs_err)
        out["rmse_to_target"] = math.sqrt(sum(e * e for e in abs_err) / len(abs_err))
        out["success@eps"] = 100.0 * sum(e <= eps for e in abs_err) / total_samples
        out["final_success@eps"] = 100 * sum(1 for i, f in enumerate(fs) if f and abs_err[i] <= eps) / total_samples

        # --- uniqueness/novelty overall ---
        valid_canon = [canonical_key(m) for m, v in zip(mols, valid, strict=False) if v]
        valid_canon = [c for c in valid_canon if c is not None]
        uniq_overall = 100.0 * len(set(valid_canon)) / n_valid if n_valid else 0.0
        novel_overall = 100.0 * len(set(valid_canon) - self.T) / n_valid if n_valid else 0.0
        out.update({"uniqueness": uniq_overall, "novelty": novel_overall})

        # --- hits-only ---
        hit_indices = [i for i, (p, t, f) in enumerate(paired) if abs(p - t) <= eps]
        if hit_indices:
            hit_keys = []
            j = -1
            for i, v in enumerate(valid):
                if not v:
                    continue
                j += 1
                if j in hit_indices and j < len(valid_canon) and valid_canon[j] is not None:
                    hit_keys.append(valid_canon[j])
            if hit_keys:
                out["uniqueness_hits"] = 100.0 * len(set(hit_keys)) / len(hit_indices)
                out["novelty_hits"] = 100.0 * len(set(hit_keys) - self.T) / len(hit_indices)

            if compute_diversity and len(hit_keys) >= 2:
                hit_mols = [mols[i] for i, v in enumerate(valid) if v][: len(hit_indices)]
                fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in hit_mols]
                sims = [
                    DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    for i in range(len(fps))
                    for j in range(i + 1, len(fps))
                ]
                if sims:
                    out["diversity_hits"] = 100.0 * (1.0 - (sum(sims) / len(sims)))

        return out

    def get_mols_and_valid_flags(self):
        return self.mols, self.valid_flags
