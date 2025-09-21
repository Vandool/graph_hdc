from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.utils.chem import eval_key_from_data


def test_eval_keys_qm9():
    for split in ["train", "valid", "test"]:
        dataset = QM9Smiles(split=split)

        count = 0
        for i, data in enumerate(dataset):
            eval_key = eval_key_from_data(data, dataset="qm9")
            count += int(eval_key == data.smiles)

        print(f"QM9 Split {split} eval key to canonical smiles proportion {count / len(dataset)}")


def test_eval_keys_zinc():
    for split in ["train", "valid", "test"]:
        dataset = ZincSmiles(split=split)

        count = 0
        for i, data in enumerate(dataset):
            eval_key = eval_key_from_data(data, dataset="zinc")
            count += int(eval_key == data.smiles)

        print(f"Zinc Split {split} eval key to canonical smiles proportion {count / len(dataset)}")
