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
from src.encoding.decoder import greedy_oracle_decoder_faster
from src.encoding.graph_encoders import HyperNet, load_or_create_hypernet
from src.encoding.oracles import Oracle
from src.utils import registery
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, GLOBAL_MODEL_PATH, DataTransformer, find_files

"""
Results
{'beam_size': 8, 'oracle_threshold': 0.5, 'strict': True, 'use_pair_feasibility': True, 'expand_on_n_anchors': 4}
/home/akaveh/Projects/kit/graph_hdc/_models/2_bah_lightning/BAH_med_qm9/models/epoch17-val0.2655.ckpt: Accuracy 0.58 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/2_bah_lightning/BAH_large_qm9/models/epoch16-val0.2740.ckpt: Accuracy 0.5 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/2_bah_lightning/BAH_base_qm9/models/epoch18-val0.2566.ckpt: Accuracy 0.6 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt: Accuracy 0.55 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/1_mlp_lightning/MLP_medium_qm9/models/epoch06-val0.2831.ckpt: Accuracy 0.46 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/1_mlp_lightning/MLP_deep_qm9/models/epoch02-val0.3406.ckpt: Accuracy 0.27 on 100 of validation set.

{'beam_size': 8, 'oracle_threshold': 0.5, 'strict': False, 'use_pair_feasibility': True, 'expand_on_n_anchors': 4}
/home/akaveh/Projects/kit/graph_hdc/_models/2_bah_lightning/BAH_med_qm9/models/epoch17-val0.2655.ckpt: Accuracy 0.58 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/2_bah_lightning/BAH_large_qm9/models/epoch16-val0.2740.ckpt: Accuracy 0.5 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/2_bah_lightning/BAH_base_qm9/models/epoch18-val0.2566.ckpt: Accuracy 0.6 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt: Accuracy 0.55 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/1_mlp_lightning/MLP_medium_qm9/models/epoch06-val0.2831.ckpt: Accuracy 0.46 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/1_mlp_lightning/MLP_deep_qm9/models/epoch02-val0.3406.ckpt: Accuracy 0.27 on 100 of validation set

{'beam_size': 16, 'oracle_threshold': 0.5, 'strict': False, 'use_pair_feasibility': False, 'expand_on_n_anchors': 8}
/home/akaveh/Projects/kit/graph_hdc/_models/2_bah_lightning/BAH_med_qm9/models/epoch17-val0.2655.ckpt: Accuracy 0.46 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/2_bah_lightning/BAH_large_qm9/models/epoch16-val0.2740.ckpt: Accuracy 0.43 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/2_bah_lightning/BAH_base_qm9/models/epoch18-val0.2566.ckpt: Accuracy 0.54 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt: Accuracy 0.56 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/1_mlp_lightning/MLP_medium_qm9/models/epoch06-val0.2831.ckpt: Accuracy 0.47 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/1_mlp_lightning/MLP_deep_qm9/models/epoch02-val0.3406.ckpt: Accuracy 0.3 on 100 of validation set.

{'beam_size': 16, 'oracle_threshold': 0.5, 'strict': False, 'use_pair_feasibility': True, 'expand_on_n_anchors': 8}
/home/akaveh/Projects/kit/graph_hdc/_models/2_bah_lightning/BAH_med_qm9/models/epoch17-val0.2655.ckpt: Accuracy 0.61 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/2_bah_lightning/BAH_large_qm9/models/epoch16-val0.2740.ckpt: Accuracy 0.62 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/2_bah_lightning/BAH_base_qm9/models/epoch18-val0.2566.ckpt: Accuracy 0.65 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt: Accuracy 0.66 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/1_mlp_lightning/MLP_medium_qm9/models/epoch06-val0.2831.ckpt: Accuracy 0.52 on 100 of validation set.
/home/akaveh/Projects/kit/graph_hdc/_models/1_mlp_lightning/MLP_deep_qm9/models/epoch02-val0.3406.ckpt: Accuracy 0.3 on 100 of validation set.
"""


# Real Oracle
def is_final_graph(G_small: nx.Graph, G_big: nx.Graph) -> bool:
    """NetworkX VF2: is `G_small` an induced, label-preserving subgraph of `G_big`?"""
    if G_small.number_of_nodes() == G_big.number_of_nodes() and G_small.number_of_edges() == G_big.number_of_edges():
        nm = lambda a, b: a["feat"] == b["feat"]
        GM = nx.algorithms.isomorphism.GraphMatcher(G_big, G_small, node_match=nm)
        return GM.subgraph_is_isomorphic()
    return False


start = 0
end = 1
batch_size = end - start
seed = 42
seed_everything(seed)
device = torch.device("cuda")
use_best_threshold = False

results: dict[str, str] = {}
# Iterate all the checkpoints
for ckpt_path in find_files(start_dir=GLOBAL_MODEL_PATH, prefixes=("epoch",), skip_substring="nvp"):
    print(f"File Name: {ckpt_path}")

    # Read the metrics from training
    evals_dir = ckpt_path.parent.parent / "evaluations"
    epoch_metrics = pd.read_parquet(evals_dir / "epoch_metrics.parquet")
    print(epoch_metrics.head(5))

    val_loss = "val_loss" if "val_loss" in epoch_metrics.columns else "val_loss_cb"
    best = epoch_metrics.loc[epoch_metrics[val_loss].idxmin()].add_suffix("_best")
    last = epoch_metrics.iloc[-1].add_suffix("_last")

    oracle_setting = {
        "beam_size": 4,
        "oracle_threshold": best["val_best_thr"] if use_best_threshold else 0.5,
        "strict": False,
        "use_pair_feasibility": True,
        "expand_on_n_anchors": 2,
    }

    ## Determine model type
    model_type: registery.ModelType = "MLP"
    if "bah" in str(ckpt_path):
        model_type = "BAH"
    elif "gin-c" in str(ckpt_path):
        model_type = "GIN-C"
    elif "gin-f" in str(ckpt_path):  # Case gin-f
        model_type = "GIN-F"

    split = "valid"
    ## Determine Dataset
    if "zinc" in str(ckpt_path):
        ds = SupportedDataset.ZINC_SMILES_HRR_7744
        dataset = ZincSmiles(split=split)
    else:  # Case qm9
        ds = SupportedDataset.QM9_SMILES_HRR_1600
        dataset = QM9Smiles(split=split)

    ## Hyper net
    hypernet: HyperNet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds.default_cfg).to(device=device)
    assert not hypernet.use_edge_features()
    assert not hypernet.use_graph_features()

    ## Classifier and Oracle
    try:
        classifier = (
            registery.retrieve_model(name=model_type)
            .load_from_checkpoint(ckpt_path, map_location="cpu", strict=True)
            .to(device)
            .eval()
        )
    except Exception as e:
        results[ckpt_path] = f"Error: {e}"
        continue
    oracle = Oracle(model=classifier, encoder=hypernet, model_type=model_type)

    ## Data loader
    dataset = dataset[start:end]
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    y = []
    correct_decoded = []
    print(f"Starting oracle decoding... for {ckpt_path}\n\tDataset: {ds.value}")
    start_t = time.perf_counter()
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
            # print(f"[{j}] Original Graph")
            # visualisations.draw_nx_with_atom_colorings(full_graph_nx, label="ORIGINAL")
            # plt.show()
            # mol_full, _ = DataTransformer.nx_to_mol(full_graph_nx)
            # display(mol_full)

            # print(f"Num Nodes {datas[g].num_nodes}")
            # print(f"Num Edges {int(datas[g].num_edges / 2)}")
            # print(f"Multiset Nodes {node_multiset.total()}")
            nx_GS: list[nx.Graph] = greedy_oracle_decoder_faster(
                node_multiset=node_multiset,
                oracle=oracle,
                full_g_h=graph_terms_hd[g],
                **oracle_setting,
            )
            # print(len(nx_GS))
            # print(nx_GS)
            nx_GS = list(filter(None, nx_GS))
            if len(nx_GS) == 0:
                y.append(0)
                # print("No Graphs encoded ...!")
                continue

            sub_g_ys = [0]
            for i, g in enumerate(nx_GS):
                is_final = is_final_graph(g, full_graph_nx)
                # print("Is final graph: ", is_final)
                # if is_final:
                # print(f"Graph Nr: {i}")
                # visualisations.draw_nx_with_atom_colorings(g, label="DECODED")
                # plt.show()
                # mol, _ = DataTransformer.nx_to_mol(g)
                # display(mol)
                # print(f"Num Atoms {mol.GetNumAtoms()}")
                # print(f"Num Bonds {mol.GetNumBonds()}")

                sub_g_ys.append(int(is_final))
            is_final_graph_ = int(sum(sub_g_ys) >= 1)
            y.append(is_final_graph_)
            if is_final_graph_:
                correct_decoded.append(j)
            sub_g_ys = []

    acc = sum(y) / len(y)
    results[ckpt_path.as_posix()] = f"Accuracy {acc} on {batch_size} of validation set."
    # print(f"Accuracy: {acc}")

    ### Save metrics
    run_metrics = {
        "path": "/".join(ckpt_path.parts[-4:]),
        "model_type": model_type,
        "dataset": ds.value,
        "time_per_sample": (time.perf_counter() - start_t) / batch_size,
        **oracle_setting,
        "oracle_acc": acc,
        **best.to_dict(),
        **last.to_dict(),
    }

    # --- new code starts here ---
    # --- save metrics to disk ---
    asset_dir = GLOBAL_ARTEFACTS_PATH / "classification"
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / "oracle_acc.parquet"
    csv_path = asset_dir / "oracle_acc.csv"

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
