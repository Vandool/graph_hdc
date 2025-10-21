from src.encoding.configs_and_constants import (
    QM9_SMILES_HRR_1600_CONFIG_F64_G1G3_CONFIG,
    QM9_SMILES_HRR_1600_CONFIG_F64_G1NG3_CONFIG,
    ZINC_SMILES_HRR_6144_G1G4_CONFIG,
)
from src.generation.analyze import analyze_terms_only
from src.generation.generation import HDCGenerator
from src.utils.utils import pick_device


def plot_sanity_plots():
    for p in [
        "nvp_QM9SmilesHRR1600F64G1G3_f4_lr0.000862736_wd0.0001_bs192_an",
        "nvp_QM9SmilesHRR1600F64G1G3_f4_lr8.69904e-5_wd0_bs288_an",
        "nvp_QM9SmilesHRR1600F64G1G3_f7_lr9.4456e-5_wd0.0003_bs448_an",
        "nvp_QM9SmilesHRR1600F64G1G3_f8_lr0.00057532_wd0.0003_bs32_an",
        "nvp_QM9SmilesHRR1600F64G1G3_f8_lr6.69953e-5_wd0.0001_bs160_an",
        "nvp_QM9SmilesHRR1600F64G1G3_f9_lr0.000179976_wd0.0003_bs288_an",
        "nvp_QM9SmilesHRR1600F64G1G3_f14_lr0.000112721_wd0.0005_bs224_an",
        "nvp_QM9SmilesHRR1600F64G1G3_f14_lr0.000132447_wd3e-6_bs160_an",
        "nvp_QM9SmilesHRR1600F64G1G3_f15_hid800_s42_lr0.000160949_wd3e-6_bs224_conNone_datSupportedDataset.QM9_SMILES_HRR_1600_F64_G1G3_epo700_expNone_hv_2_hv_1600_is_0_res0_smf4_smi1_smw10_use1_vsaVSAModel.HRR_an",
        "nvp_QM9SmilesHRR1600F64G1G3_f15_lr0.000160949_wd3e-6_bs224_an",
        "nvp_QM9SmilesHRR1600F64G1G3_f15_lr6.29685e-5_wd3e-6_bs128_an",
        "nvp_QM9SmilesHRR1600F64G1G3_f16_hid400_s42_lr0.000154612_wd3e-6_bs32_conNone_datSupportedDataset.QM9_SMILES_HRR_1600_F64_G1G3_epo700_expNone_hv_2_hv_1600_is_0_res0_smf4_smi1_smw10_use1_vsaVSAModel.HRR_an",
        "nvp_QM9SmilesHRR1600F64G1G3_f16_hid800_s42_lr0.000430683_wd3e-6_bs512_conNone_datSupportedDataset.QM9_SMILES_HRR_1600_F64_G1G3_epo700_expNone_hv_2_hv_1600_is_0_res0_smf4_smi1_smw10_use1_vsaVSAModel.HRR_an",
        "nvp_QM9SmilesHRR1600F64G1G3_f16_lr0.000525421_wd0.0005_bs256_an",
    ]:
        print(p)
        try:
            generator = HDCGenerator(
                gen_model_hint=p, ds_config=QM9_SMILES_HRR_1600_CONFIG_F64_G1G3_CONFIG, device=pick_device()
            )
        except Exception:
            print(f"[FAILED] {p}")
            continue

        samples = generator.get_raw_samples(n_samples=1000)
        analyze_terms_only(terms=samples, name=p)

        for p in [
            "nvp_QM9SmilesHRR1600F64G1NG3_f4_lr0.000862736_wd0.0001_bs192_an",
            "nvp_QM9SmilesHRR1600F64G1NG3_f4_lr8.69904e-5_wd0_bs288_an",
            "nvp_QM9SmilesHRR1600F64G1NG3_f7_lr9.4456e-5_wd0.0003_bs448_an",
            "nvp_QM9SmilesHRR1600F64G1NG3_f8_hid800_s42_lr9.4456e-5_wd0.0003_bs448_conNone_datSupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3_epo700_expNone_hv_2_hv_1600_is_0_res0_smf4_smi1_smw10_use1_vsaVSAModel.HRR_an",
            "nvp_QM9SmilesHRR1600F64G1NG3_f8_lr6.69953e-5_wd0.0001_bs160_an",
            "nvp_QM9SmilesHRR1600F64G1NG3_f14_hid1600_s42_lr0.000525421_wd0.0005_bs256_conNone_datSupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3_epo700_expNone_hv_2_hv_1600_is_0_res0_smf4_smi1_smw10_use1_vsaVSAModel.HRR_an",
            "nvp_QM9SmilesHRR1600F64G1NG3_f14_lr0.000112721_wd0.0005_bs224_an",
            "nvp_QM9SmilesHRR1600F64G1NG3_f15_hid1600_s42_lr0.0004818_wd0.0005_bs288_conNone_datSupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3_epo700_expNone_hv_2_hv_1600_is_0_res0_smf4_smi1_smw10_use1_vsaVSAModel.HRR_an",
            "nvp_QM9SmilesHRR1600F64G1NG3_f16_hid400_s42_lr0.000862736_wd0.0001_bs192_conNone_datSupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3_epo700_expNone_hv_2_hv_1600_is_0_res0_smf4_smi1_smw10_use1_vsaVSAModel.HRR_an",
            "nvp_QM9SmilesHRR1600F64G1NG3_f16_hid1600_s42_lr0.000221865_wd0.0005_bs32_conNone_datSupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3_epo700_expNone_hv_2_hv_1600_is_0_res0_smf4_smi1_smw10_use1_vsaVSAModel.HRR_an",
            "nvp_QM9SmilesHRR1600F64G1NG3_f16_lr0.000525421_wd0.0005_bs256_an",
        ]:
            print(p)
        try:
            generator = HDCGenerator(
                gen_model_hint=p, ds_config=QM9_SMILES_HRR_1600_CONFIG_F64_G1NG3_CONFIG, device=pick_device()
            )
        except Exception:
            print(f"[FAILED] {p}")
            continue

        samples = generator.get_raw_samples(n_samples=1000)
        analyze_terms_only(terms=samples, name=p)

        for p in [
            "nvp_ZincSmilesHRR6144F64G1G4_f4_lr0.000862736_wd0.0001_bs192_an",
            "nvp_ZincSmilesHRR6144F64G1G4_f7_lr9.4456e-5_wd0.0003_bs448_an",
            "nvp_ZincSmilesHRR6144F64G1G4_f16_lr0.000525421_wd0.0005_bs256_an",
            "nvp_ZincSmilesHRR6144F64G1G4_lr0.00057532_wd0.0003_bs32_an",
            "nvp_ZincSmilesHRR6144F64G1G4_lr0.000112721_wd0.0005_bs224_an",
            "nvp_ZincSmilesHRR6144F64G1G4_lr0.000132447_wd3e-6_bs160_an",
            "nvp_ZincSmilesHRR6144F64G1G4_lr0.000525421_wd0.0005_bs256_an",
            "nvp_ZincSmilesHRR6144F64G1G4_lr0.000862736_wd0.0001_bs192_an",
            "nvp_ZincSmilesHRR6144F64G1G4_lr5.11626e-5_wd0.0001_bs160_an",
            "nvp_ZincSmilesHRR6144F64G1G4_lr6.69953e-5_wd0.0001_bs160_an",
            "nvp_ZincSmilesHRR6144F64G1G4_lr8.69904e-5_wd0_bs288_an",
            "nvp_ZincSmilesHRR6144F64G1G4_lr9.4456e-5_wd0.0003_bs448_an",
        ]:
            print(p)
        try:
            generator = HDCGenerator(gen_model_hint=p, ds_config=ZINC_SMILES_HRR_6144_G1G4_CONFIG, device=pick_device())
        except Exception:
            print(f"[FAILED] {p}")
            continue

        samples = generator.get_raw_samples(n_samples=1000)
        analyze_terms_only(terms=samples, name=p)
