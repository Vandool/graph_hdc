from pprint import pprint

from src.normalizing_flow.config import SpiralFlowConfig, get_flow_cli_args


def run_experiment(cfg: SpiralFlowConfig):
    print("Running experiment")
    print(pprint(cfg.__dict__, indent=2))





if __name__ == '__main__':
    run_experiment(get_flow_cli_args())