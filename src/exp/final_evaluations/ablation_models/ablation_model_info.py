import json
import os

import torch

from src.generation.generation import get_model_path
from src.utils import registery
from src.utils.registery import get_model_type

ablation_model_hints = [
    "nvp_comp_QM9SmilesHRR256F64G1NG3_f6_hid1536_lr0.001_wd0.000417855_bs192_np1_smf6.5_smi2.2_smw16_an_dim",
    "nvp_comp_QM9SmilesHRR256F64G1NG3_f16_hid512_lr0.001_wd1.31526e-6_bs384_np1_smf6.5_smi2.2_smw16_an_dim",
    "nvp_v3_comp_QM9SmilesHRR256F64G1NG3_f16_lr0.000190993_wd0.0003491_bs96_hid1792_nhl3_np1_smf6.5_smi2.2_smw16_an",
    "nvp_v3_comp_QM9SmilesHRR256F64G1NG3_f6_lr0.000557295_wd1.022e-5_bs288_hid2048_nhl2_np0_smf6.5_smi2.2_smw16_an_dim",
    "sf_hpo_QM9SmilesHRR256F64G1NG3_num6_num1024_lr0.000108339_wd2.51289e-7_bs128_dro0.2_num12_num6_an",
    "fm_comp_QM9SmilesHRR256F64G1NG3_s42_lr8.76102e-5_wd1e-6_bs128_hid1536_nhl8_np1_tim32_an",
    "R1_nvp_QM9SmilesHRR1600F64G1NG3_f16_hid400_lr0.000345605_wd3e-6_bs160_smf6.5_smi2.2_smw16_an",
    "nvp_QM9SmilesHRR1600F64G1G3_f9_hid800_lr0.000167241_wd3e-6_bs128_smf6.5_smi2.2_smw16_an",
    "sf_hpo_ZincSmilesHRR256F645G1NG4_num8_num768_lr0.000412321_wd1.5703e-5_bs256_dro0.3_num4_num2_an",
    "fm_comp_ZincSmilesHRR256F645G1NG4_s42_lr0.000607711_wd3.04221e-5_bs512_hid2048_nhl4_np1_tim32_an",
    "nvp_v3_comp_ZincSmilesHRR256F645G1NG4_f8_lr0.000539046_wd0.001_bs224_hid1536_nhl2_np1_smf7_smi2.5_smw17_an",
    "nvp_comp_ZincSmilesHRR256F645G1NG4_f9_hid1024_lr0.000854811_wd5.72439e-5_bs224_np1_smf7_smi2.5_smw17_an",
    "nvp_ZincSmilesHRR1024F645G1NG4_f11_hid1024_lr0.000343816_wd3e-6_bs160_smf7_smi2.5_smw17_an",
    "nvp_ZincSmilesHRR2048F645G1NG4_f14_hid2048_lr6.25028e-5_wd3e-6_bs128_smf7_smi2.5_smw17_an",
    "nvp_ZincSmilesHRR5120F64G1G4_lr6.69953e-5_wd0.0001_bs160_an",
]


def count_parameters(model) -> float:
    """Count total parameters and return size in millions."""
    total_params = sum(p.numel() for p in model.parameters())
    return round(total_params / 1e6)  # Convert to millions


def normalize_value(value):
    """Recursively normalize a value to be JSON-serializable."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        # Handle objects with __dict__ (e.g., custom classes, namespaces)
        return normalize_value(vars(value))
    if hasattr(value, "item"):
        # Handle numpy/torch scalars
        return value.item()
    if hasattr(value, "tolist"):
        # Handle numpy/torch arrays
        return value.tolist()
    # Fallback to string representation
    return str(value)


def normalize_hparams(hparams) -> dict:
    """Recursively normalize hparams to be JSON-serializable."""
    if hasattr(hparams, "items"):
        # Dict-like object
        return {str(k): normalize_value(v) for k, v in hparams.items()}
    if hasattr(hparams, "__dict__"):
        # Namespace or custom object
        return {str(k): normalize_value(v) for k, v in vars(hparams).items()}
    return normalize_value(hparams)


if __name__ == "__main__":
    final_results = {}
    for gen_hint in ablation_model_hints:
        gen_ckpt_path = get_model_path(hint=gen_hint)
        print(f"Generator Checkpoint: {gen_ckpt_path}")
        model_type = get_model_type(gen_ckpt_path)
        print(f"Model Type: {model_type}")
        assert model_type is not None, f"Model type not found for {gen_ckpt_path}"

        gen_model = (
            registery.retrieve_model(model_type)
            .load_from_checkpoint(gen_ckpt_path, map_location="cpu", strict=True)
            .to(torch.device("cuda"))
        )

        model_size_m = count_parameters(gen_model)

        model_info = {
            # "hparams": normalize_hparams(gen_model.hparams),  # Convert to dict for JSON serialization
            "model_size_M": model_size_m,
            "model_repr": repr(gen_model).replace("\n", "").replace(" ", ""),
        }
        final_results[gen_hint] = model_info

        print(f"Model size: {model_size_m:.6f} M parameters")
        # print(gen_model)
        print("-" * 80)

    # Save results to JSON in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "model_repr.json")

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)  # default=str handles non-serializable types

    print(f"\nResults saved to: {output_path}")
