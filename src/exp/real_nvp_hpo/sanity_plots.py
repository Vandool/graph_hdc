from src.encoding.configs_and_constants import (
    QM9_SMILES_HRR_1600_CONFIG_F64_G1G3_CONFIG,
    QM9_SMILES_HRR_1600_CONFIG_F64_G1NG3_CONFIG,
    ZINC_SMILES_HRR_6144_G1G4_CONFIG,
)
from src.generation.analyze import analyze_terms_only
from src.generation.generation import HDCGenerator
from src.utils.utils import GLOBAL_MODEL_PATH, find_files, pick_device


def plot_sanity_plots():
    ds_configs_to_try = [
        QM9_SMILES_HRR_1600_CONFIG_F64_G1G3_CONFIG,
        QM9_SMILES_HRR_1600_CONFIG_F64_G1NG3_CONFIG,
        ZINC_SMILES_HRR_6144_G1G4_CONFIG,
    ]
    for p in find_files(start_dir=GLOBAL_MODEL_PATH / "0_real_nvp", prefixes=("last",), desired_ending=".ckpt"):
        name = p.parent.parent.name
        if (ds_config := next((d for d in ds_configs_to_try if d.name in name), None)) is None:
            print(f"[SKIPPED] {p}")
            continue
        try:
            generator = HDCGenerator(gen_model_hint=p, ds_config=ds_config, device=pick_device())
        except Exception:
            print(f"[FAILED] {p}")
            continue

        samples = generator.get_raw_samples(n_samples=1000)
        analyze_terms_only(terms=samples, name=p)


if __name__ == "__main__":
    plot_sanity_plots()
