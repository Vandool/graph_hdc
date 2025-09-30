import time

import networkx as nx
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from torch_geometric.loader import DataLoader
from torchhd import HRRTensor

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import SupportedDataset
from src.encoding.decoder import (
    greedy_oracle_decoder_voter_oracle,
    is_induced_subgraph_by_features,
)
from src.encoding.graph_encoders import HyperNet, load_or_create_hypernet
from src.encoding.oracles import SimpleVoterOracle
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, GLOBAL_MODEL_PATH, DataTransformer, find_files


# Real Oracle
def is_final_graph(G_small: nx.Graph, G_big: nx.Graph) -> bool:
    """NetworkX VF2: is `G_small` an induced, label-preserving subgraph of `G_big`?"""
    if G_small.number_of_nodes() == G_big.number_of_nodes() and G_small.number_of_edges() == G_big.number_of_edges():
        return is_induced_subgraph_by_features(G_small, G_big)
    return False


start = 0
end = 100
batch_size = end - start
seed = 42
seed_everything(seed)
device = torch.device("cpu")
use_best_threshold = True

results: dict[str, str] = {}
# Iterate all the checkpoints
files = list(find_files(start_dir=GLOBAL_MODEL_PATH, prefixes=("epoch",), skip_substrings=("nvp", "zinc")))
print(f"Found {len(files)} checkpoints.")
for model_paths in [
    [
        GLOBAL_MODEL_PATH / "1_gin/gin-f_baseline_qm9_resume/models/epoch10-val0.3359.ckpt",
        GLOBAL_MODEL_PATH / "1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt",
        GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_large_qm9/models/epoch16-val0.2740.ckpt",
        # GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_med_qm9/models/epoch19-val0.2648.ckpt",
        # GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_base_qm9_v2/models/epoch23-val0.2772.ckpt",
        # GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_larger_qm9/models/epoch16-val0.2949.ckpt",
    ],
    [
        GLOBAL_MODEL_PATH / "1_gin/gin-f_baseline_qm9_resume/models/epoch10-val0.3359.ckpt",
        GLOBAL_MODEL_PATH / "1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt",
        # GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_large_qm9/models/epoch16-val0.2740.ckpt",
        GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_med_qm9/models/epoch19-val0.2648.ckpt",
        # GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_base_qm9_v2/models/epoch23-val0.2772.ckpt",
        # GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_larger_qm9/models/epoch16-val0.2949.ckpt",
    ],
    [
        GLOBAL_MODEL_PATH / "1_gin/gin-f_baseline_qm9_resume/models/epoch10-val0.3359.ckpt",
        GLOBAL_MODEL_PATH / "1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt",
        # GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_large_qm9/models/epoch16-val0.2740.ckpt",
        # GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_med_qm9/models/epoch19-val0.2648.ckpt",
        GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_base_qm9_v2/models/epoch23-val0.2772.ckpt",
        # GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_larger_qm9/models/epoch16-val0.2949.ckpt",
    ],
    [
        GLOBAL_MODEL_PATH / "1_gin/gin-f_baseline_qm9_resume/models/epoch10-val0.3359.ckpt",
        GLOBAL_MODEL_PATH / "1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt",
        # GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_large_qm9/models/epoch16-val0.2740.ckpt",
        # GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_med_qm9/models/epoch19-val0.2648.ckpt",
        # GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_base_qm9_v2/models/epoch23-val0.2772.ckpt",
        GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_larger_qm9/models/epoch16-val0.2949.ckpt",
    ],
]:
    oracle_setting = {
        "beam_size": 512,
        "strict": False,
        "use_pair_feasibility": True,
        "expand_on_n_anchors": 12,
    }

    split = "valid"
    ## Determine Dataset
    if "zinc" in str(model_paths[0]).lower():
        ds = SupportedDataset.ZINC_SMILES_HRR_7744
        dataset = ZincSmiles(split=split)
        dataset_base = "zinc"
    else:  # Case qm9
        ds = SupportedDataset.QM9_SMILES_HRR_1600
        dataset = QM9Smiles(split=split)
        dataset_base = "qm9"

    ## Hyper net
    hypernet: HyperNet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds.default_cfg).to(device=device)
    assert not hypernet.use_edge_features()
    assert not hypernet.use_graph_features()

    oracle = SimpleVoterOracle(model_paths=model_paths, encoder=hypernet, device=device)

    ## Data loader
    dataset = dataset[start:end]
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    y = []
    correct_decoded = []
    start_t = time.perf_counter()
    print(f"Model Paths: {model_paths}")
    for i, batch in enumerate(dataloader):
        # Encode the whole graph in one HV
        encoded_data = hypernet.forward(batch)
        node_term = encoded_data["node_terms"]
        graph_term = encoded_data["graph_embedding"]

        graph_terms_hd = graph_term.as_subclass(HRRTensor)

        ground_truth_counters = {}
        datas = batch.to_data_list()
        for j, g in enumerate(range(batch_size)):
            # print("================================================")
            full_graph_nx = DataTransformer.pyg_to_nx(data=datas[g])
            node_multiset = DataTransformer.get_node_counter_from_batch(batch=g, data=batch)
            nx_GS, _ = greedy_oracle_decoder_voter_oracle(
                node_multiset=node_multiset,
                oracle=oracle,
                full_g_h=graph_terms_hd[g],
                **oracle_setting,
            )
            nx_GS = list(filter(None, nx_GS))
            if len(nx_GS) == 0:
                y.append(0)
                continue

            sub_g_ys = [0]
            for i, g in enumerate(nx_GS):
                is_final = is_final_graph(g, full_graph_nx)

                sub_g_ys.append(int(is_final))
            is_final_graph_ = int(sum(sub_g_ys) >= 1)
            # print(is_final_graph_)
            y.append(is_final_graph_)
            if is_final_graph_:
                correct_decoded.append(j)
            sub_g_ys = []

    acc = sum(y) / len(y)
    # print(f"Accuracy: {acc}")

    ### Save metrics
    run_metrics = {
        "path": f"{model_paths!s}",
        "model_type": f"OracleSimpleVoter-{len(model_paths)}-0.5Threshold",
        "back_tracing": True,
        "dataset": ds.value,
        "num_eval_samples": batch_size,
        "time_per_sample": (time.perf_counter() - start_t) / batch_size,
        **oracle_setting,
        "oracle_acc": acc,
    }

    # --- new code starts here ---
    # --- save metrics to disk ---
    asset_dir = GLOBAL_ARTEFACTS_PATH / "classification"
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / f"oracle_acc_voter_{dataset_base}.parquet"
    csv_path = asset_dir / f"oracle_acc_voter_{dataset_base}.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # append current row
    new_row = pd.DataFrame([run_metrics])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # write back out
    metrics_df.to_parquet(parquet_path, index=False)
    metrics_df.to_csv(csv_path, index=False)


print(oracle_setting)
for k, v in results.items():
    print(f"{k}: {v}")
