from pathlib import Path

from src.encoding.configs_and_constants import BaseDataset, SupportedDataset
from src.utils.utils import GLOBAL_BEST_MODEL_PATH, find_files

# Models ordered by generative performance
GENERATOR_REGISTRY = {
    SupportedDataset.QM9_SMILES_HRR_256_F64_G1NG3: [
        "nvp_v3_comp_QM9SmilesHRR256F64G1NG3_f13_lr0.00018206_wd0.000158329_bs192_hid1280_nhl4_np1_smf6.5_smi2.2_smw16_an",
        "nvp_v3_comp_QM9SmilesHRR256F64G1NG3_f11_lr0.000172439_wd0.000501115_bs96_hid2048_nhl4_np1_smf6.5_smi2.2_smw16_an",
        "nvp_v3_comp_QM9SmilesHRR256F64G1NG3_f12_lr0.00018451_wd0.000294167_bs128_hid2048_nhl4_np1_smf6.5_smi2.2_smw16_an",
        "nvp_v3_comp_QM9SmilesHRR256F64G1NG3_f12_lr0.000183872_wd0.000202358_bs96_hid1792_nhl4_np1_smf6.5_smi2.2_smw16_an",
    ],
    SupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3: [
        "R1_nvp_QM9SmilesHRR1600F64G1NG3_f16_lr0.000525421_wd0.0005_bs256_an",
        "R1_nvp_QM9SmilesHRR1600F64G1NG3_f15_hid1600_s42_lr0.0004818_wd0.0005_bs288",
        "R1_nvp_QM9SmilesHRR1600F64G1NG3_f16_hid400_lr0.000345605_wd3e-6_bs160_smf6.5_smi2.2_smw16_an",
        "R1_nvp_QM9SmilesHRR1600F64G1NG3_f16_hid1600_s42_lr0.000221865_wd0.0005_bs32",
    ],
    SupportedDataset.ZINC_SMILES_HRR_256_F64_5G1NG4: [
        "nvp_v3_comp_ZincSmilesHRR256F645G1NG4_f8_lr0.000539046_wd0.001_bs224_hid1536_nhl2_np1_smf7_smi2.5_smw17_an",
        # "nvp_v3_comp_ZincSmilesHRR256F645G1NG4_f8_lr0.00054266_wd0.000716922_bs192_hid1536_nhl2_np1_smf7_smi2.5_smw17_an",
        "nvp_v3_comp_ZincSmilesHRR256F645G1NG4_f8_lr0.000531954_wd0.000587484_bs192_hid1280_nhl2_np1_smf7_smi2.5_smw17_an",
        # "nvp_v3_comp_ZincSmilesHRR256F645G1NG4_f10_lr0.000571578_wd0.000438479_bs320_hid1024_nhl2_np1_smf7_smi2.5_smw17_an",
    ],
    SupportedDataset.ZINC_SMILES_HRR_1024_F64_5G1NG4: [
        # "nvp_ZincSmilesHRR1024F645G1NG4_f11_hid1024_lr0.000313799_wd3e-6_bs96_smf7_smi2.5_smw17_an",
        "nvp_ZincSmilesHRR1024F645G1NG4_f11_hid1024_lr0.000343816_wd3e-6_bs160_smf7_smi2.5_smw17_an",
        # "nvp_ZincSmilesHRR1024F645G1NG4_f11_hid1024_lr0.000640292_wd3e-6_bs192_smf7_smi2.5_smw17_an",
    ],
    SupportedDataset.ZINC_SMILES_HRR_2048_F64_5G1NG4: [
        "nvp_ZincSmilesHRR2048F645G1NG4_f10_hid2048_lr0.000119405_wd0.0005_bs128_smf7_smi2.5_smw17_an",
        "nvp_ZincSmilesHRR2048F645G1NG4_f11_hid2048_lr6.29685e-5_wd3e-6_bs128_smf7_smi2.5_smw17_an",
        "nvp_ZincSmilesHRR2048F645G1NG4_f6_hid2048_lr0.000112721_wd0.0005_bs224_smf7_smi2.5_smw17_an",
        "nvp_ZincSmilesHRR2048F645G1NG4_f8_hid1024_lr9.4456e-5_wd0.0003_bs448_smf7_smi2.5_smw17_an",
        "nvp_ZincSmilesHRR2048F645G1NG4_f10_hid1024_lr6.69953e-5_wd0.0001_bs160_smf7_smi2.5_smw17_an",
        "nvp_ZincSmilesHRR2048F645G1NG4_f13_hid2048_lr0.000121995_wd3e-6_bs128_smf7_smi2.5_smw17_an",
    ],
    SupportedDataset.ZINC_SMILES_HRR_5120_F64_G1G3: [
        "nvp_ZincSmilesHRR5120F64G1G4_lr6.69953e-5_wd0.0001_bs160_an",
    ],
}

REGRESSOR_REGISTRY = {
    SupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3: {
        "logp": [
            "pr_logp_QM9SmilesHRR1600F64G1NG3_h512-256-128_actgelu_nmnone_dp0.010168_bs320_lr0.000355578_wd1e-5_dep3_h1512_h2256_h3128_h4160"
        ],
        "qed": [
            "pr_qed_QM9SmilesHRR1600F64G1NG3_h768-896-192_actsilu_nmln_dp0.0397431_bs288_lr8.69904e-5_wd0_dep4_h1768_h2896_h3192_h496"
        ],
        "sa_score": [
            "pr_sa_score_QM9SmilesHRR1600F64G1NG3_h768-256-64_actlrelu_nmln_dp0.0600357_bs352_lr0.000649563_wd0.0005_dep4_h1768_h2256_h364_h4160"
        ],
        "max_ring_size": [],
    },
    SupportedDataset.ZINC_SMILES_HRR_1024_F64_5G1NG4: {
        "logp": [
            "pr_logp_ZincSmilesHRR1024F645G1NG4_h768-384_actsilu_nmnone_dp0.121133_bs160_lr9.65386e-5_wd3e-5_dep2_h1768_h2384_h364_h4160"
        ],
        "qed": [
            "pr_qed_ZincSmilesHRR1024F645G1NG4_h512-512-256_actsilu_nmln_dp0.00684795_bs128_lr5.42763e-5_wd0_dep4_h1512_h2512_h3256_h432"
        ],
        "sa_score": [],
        "max_ring_size": [],
    },
    SupportedDataset.ZINC_SMILES_HRR_2048_F64_5G1NG4: {
        "logp": [
            "pr_logp_ZincSmilesHRR2048F645G1NG4_h1024-640-512_actsilu_nmnone_dp0.00819581_bs512_lr0.000251627_wd1e-5_dep3_h11024_h2640_h3512_h464"
        ],
        "qed": [],
        "sa_score": [],
        "max_ring_size": [],
    },
}


DECODER_SETTINGS = {
    "qm9": {
        "iteration_budget": 3,
        "max_graphs_per_iter": 1024,
        "top_k": 10,
        "sim_eps": 0.0001,
        "early_stopping": True,
        "prefer_smaller_corrective_edits": False,
        "fallback_decoder_settings": {
            "initial_limit": 2048,
            "limit": 1024,
            "beam_size": 1024,
            "pruning_method": "cos_sim",
            "use_size_aware_pruning": True,
            "use_one_initial_population": False,
            "use_g3_instead_of_h3": False,
            "validate_ring_structure": False,
            "use_modified_graph_embedding": True,
            "random_sample_ratio": 0.0,
        },
    },
    "zinc": {
        "iteration_budget": 20,
        "max_graphs_per_iter": 512,
        "top_k": 10,
        "sim_eps": 0.0001,
        "early_stopping": True,
        "prefer_smaller_corrective_edits": False,
        "fallback_decoder_settings": {
            "initial_limit": 2048,
            "limit": 256,
            "beam_size": 96,
            "pruning_method": "cos_sim",
            "use_size_aware_pruning": True,
            "use_one_initial_population": False,
            "use_g3_instead_of_h3": False,
            "validate_ring_structure": True,
            "use_modified_graph_embedding": True,
            "random_sample_ratio": 0.0,
        },
    },
}

DATASET_STATS: dict[BaseDataset, dict] = {
    "zinc": {
        "dataset": "train",
        "edges": {"max": 44, "mean": 24.906559217493673, "median": 25.0, "min": 5, "std": 5.2931777992057825},
        "logp": {"max": 8, "mean": 2.457799800788871, "median": 2.60617995262146, "min": -6, "std": 1.4334213538628746},
        "qed": {
            "max": 0,
            "mean": 0.7318043454311982,
            "median": 0.7625745534896851,
            "min": 0,
            "std": 0.1386172168886604,
        },
        "sa_score": {
            "max": 7,
            "mean": 3.053571513973725,
            "median": 2.8936047554016113,
            "min": 1,
            "std": 0.8347014235153748,
        },
        "max_ring_size": {"max": 24, "mean": 5.952266023062483, "median": 6.0, "min": 0, "std": 0.522215432228299},
        "node_type_distribution": {
            (0, 0, 0, 0, 0): 11241,
            (1, 0, 0, 1, 0): 1147,
            (1, 0, 0, 2, 0): 2770,
            (1, 0, 0, 3, 0): 388718,
            (1, 0, 2, 2, 0): 1,
            (1, 1, 0, 0, 0): 14353,
            (1, 1, 0, 1, 0): 18285,
            (1, 1, 0, 1, 1): 1133197,
            (1, 1, 0, 2, 0): 290265,
            (1, 1, 0, 2, 1): 538419,
            (1, 1, 2, 1, 1): 2,
            (1, 2, 0, 0, 0): 203257,
            (1, 2, 0, 0, 1): 879101,
            (1, 2, 0, 1, 0): 80985,
            (1, 2, 0, 1, 1): 158500,
            (1, 3, 0, 0, 0): 21514,
            (1, 3, 0, 0, 1): 23120,
            (2, 0, 0, 0, 0): 37828,
            (3, 0, 0, 0, 0): 69973,
            (4, 0, 0, 0, 0): 795,
            (5, 0, 0, 0, 0): 11984,
            (5, 0, 0, 1, 0): 326,
            (5, 0, 0, 2, 0): 15387,
            (5, 0, 1, 2, 0): 234,
            (5, 0, 1, 3, 0): 5970,
            (5, 0, 2, 0, 0): 44,
            (5, 0, 2, 1, 0): 10,
            (5, 1, 0, 0, 0): 6375,
            (5, 1, 0, 0, 1): 164699,
            (5, 1, 0, 1, 0): 150223,
            (5, 1, 0, 1, 1): 28380,
            (5, 1, 1, 0, 0): 47,
            (5, 1, 1, 1, 0): 713,
            (5, 1, 1, 1, 1): 6224,
            (5, 1, 1, 2, 0): 11957,
            (5, 1, 1, 2, 1): 3237,
            (5, 1, 2, 0, 0): 1000,
            (5, 1, 2, 0, 1): 290,
            (5, 2, 0, 0, 0): 25423,
            (5, 2, 0, 0, 1): 149778,
            (5, 2, 1, 0, 0): 9762,
            (5, 2, 1, 0, 1): 893,
            (5, 2, 1, 1, 0): 7994,
            (5, 2, 1, 1, 1): 20705,
            (5, 3, 1, 0, 0): 38,
            (5, 3, 1, 0, 1): 64,
            (6, 0, 0, 0, 0): 307478,
            (6, 0, 0, 1, 0): 26319,
            (6, 0, 1, 1, 0): 4,
            (6, 0, 2, 0, 0): 21513,
            (6, 1, 0, 0, 0): 85914,
            (6, 1, 0, 0, 1): 67035,
            (6, 1, 1, 0, 0): 1,
            (6, 1, 1, 0, 1): 10,
            (7, 1, 0, 1, 0): 1,
            (7, 1, 0, 2, 0): 1,
            (7, 2, 0, 0, 0): 1,
            (7, 2, 0, 0, 1): 2,
            (7, 2, 0, 1, 0): 1,
            (7, 2, 1, 1, 0): 1,
            (7, 3, 0, 0, 0): 84,
            (7, 3, 0, 0, 1): 22,
            (7, 3, 0, 1, 1): 1,
            (7, 3, 1, 0, 1): 1,
            (7, 4, 0, 0, 1): 1,
            (8, 0, 0, 0, 0): 3901,
            (8, 0, 0, 1, 0): 420,
            (8, 0, 1, 1, 0): 2,
            (8, 0, 2, 0, 0): 397,
            (8, 1, 0, 0, 0): 17105,
            (8, 1, 0, 0, 1): 42320,
            (8, 1, 1, 0, 0): 1,
            (8, 1, 1, 0, 1): 1,
            (8, 2, 0, 0, 0): 1688,
            (8, 2, 0, 0, 1): 266,
            (8, 2, 1, 0, 0): 1,
            (8, 2, 1, 0, 1): 1,
            (8, 3, 0, 0, 0): 21559,
            (8, 3, 0, 0, 1): 3042,
        },
        "nodes": {"max": 38, "mean": 23.154851348341673, "median": 23.0, "min": 6, "std": 4.507367809672409},
        "num_graphs": 220011,
        "total_node_types": 79,
    },
    "qm9": {
        "dataset": "train",
        "edges": {"max": 14, "mean": 9.396537655935868, "median": 9.0, "min": 0, "std": 1.1682700245850308},
        "logp": {
            "max": 3,
            "mean": 0.30487121410781287,
            "median": 0.27810001373291016,
            "min": -5,
            "std": 0.9661956285099275,
        },
        "qed": {
            "max": 0,
            "mean": 0.46682894885217113,
            "median": 0.47281739115715027,
            "min": 0,
            "std": 0.07255290817165666,
        },
        "sa_score": {
            "max": 7,
            "mean": 4.2913322150323685,
            "median": 4.3130269050598145,
            "min": 1,
            "std": 0.9624421465089988,
        },
        "max_ring_size": {"max": 9, "mean": 4.210642754397329, "median": 5.0, "min": 0, "std": 1.7229798089811252},
        "node_type_distribution": {
            (0, 0, 0, 0): 2,
            (0, 0, 0, 1): 12653,
            (0, 0, 0, 2): 92,
            (0, 0, 0, 3): 102500,
            (0, 0, 0, 4): 1,
            (0, 0, 1, 2): 1,
            (0, 0, 2, 0): 1,
            (0, 0, 2, 2): 1,
            (0, 1, 0, 0): 37875,
            (0, 1, 0, 1): 75277,
            (0, 1, 0, 2): 204962,
            (0, 1, 1, 1): 24,
            (0, 1, 2, 1): 15,
            (0, 2, 0, 0): 85894,
            (0, 2, 0, 1): 177453,
            (0, 2, 1, 0): 181,
            (0, 2, 2, 0): 2,
            (0, 3, 0, 0): 54878,
            (1, 0, 0, 0): 15798,
            (1, 0, 0, 1): 9588,
            (1, 0, 0, 2): 12143,
            (1, 0, 0, 3): 65,
            (1, 0, 1, 3): 72,
            (1, 0, 2, 0): 13,
            (1, 0, 2, 1): 6,
            (1, 1, 0, 0): 32284,
            (1, 1, 0, 1): 31452,
            (1, 1, 1, 2): 89,
            (1, 1, 2, 0): 13,
            (1, 2, 0, 0): 22373,
            (1, 2, 1, 0): 132,
            (1, 2, 1, 1): 30,
            (2, 0, 0, 0): 50131,
            (2, 0, 0, 1): 46504,
            (2, 0, 1, 0): 1,
            (2, 0, 2, 0): 490,
            (2, 1, 0, 0): 69733,
            (2, 1, 1, 0): 10,
            (3, 0, 0, 0): 2931,
        },
        "nodes": {"max": 9, "mean": 8.796086777311384, "median": 9.0, "min": 1, "std": 0.5088587487090748},
        "num_graphs": 118879,
        "total_node_types": 39,
    },
}


def get_pr_path(hint: str) -> Path | None:
    paths = find_files(
        start_dir=GLOBAL_BEST_MODEL_PATH / "pr",
        prefixes=("epoch",),
        desired_ending=".ckpt",
    )
    for p in paths:
        if hint in str(p):
            return p
    return None
