
from torch_geometric.loader import DataLoader

from src.datasets.zinc_smiles_generation import ZincSmiles

if __name__ == "__main__":
    train_dataset = ZincSmiles(split="train", enc_suffix="HRR7744")[:8]
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch)
        print(batch.node_terms)
        print(batch.graph_terms)
