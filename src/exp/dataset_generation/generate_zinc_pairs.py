from src.datasets.zinc_pairs import ZincPairs
from src.datasets.zinc_smiles_generation import ZincSmiles

if __name__ == '__main__':
    train_ds = ZincSmiles(split="train")
    valid_ds = ZincSmiles(split="valid")
    test_ds = ZincSmiles(split="test")

    test_pairs = ZincPairs(base_dataset=test_ds, split="test")
    valid_pairs = ZincPairs(base_dataset=valid_ds, split="valid")
    train_pairs = ZincPairs(base_dataset=train_ds, split="train")
