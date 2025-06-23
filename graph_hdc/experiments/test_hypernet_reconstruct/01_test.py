from pycomex.functional.experiment import Experiment
from pycomex.util import file_namespace, folder_path

# :param FILE_PATH:
#       String a path for saving the file
FILE_PATH: str = "/home/ka/ka_iti/ka_zi9629/projects/graph_hdc/graph_hdc/experiments/hypernet_reconstruct/results/default"


experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

import  logging
logger = logging.getLogger(__name__)

import os

@experiment.hook("create_file", replace=False, default=True)
def create_empty_file(e: Experiment, path: str) -> None:
    # Ensure the parent directory exists
    path = f"{path}/text.txt"
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    # Create (or truncate) the file
    with open(path, 'w'):
        pass



@experiment.hook("print_cuda")
def print_cuda_status(e: Experiment) -> str:
    logger.log(level=20, msg="Logging inside the function!!!")
    try:
        import torch
    except ImportError:
        return "PyTorch is not installed in this environment."

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        res = f"CUDA is available. Detected {count} GPU device{'s' if count != 1 else ''}."
    else:
        res = f"CUDA is not available."

    return res


@experiment
def experiment(e: Experiment):
    e.log(f"{FILE_PATH=}")
    e.log(f"{e.path=}")
    e.log(f"{e.parameters=}")
    e.log(f"{e.parameters['FILE_PATH']}")
    e.log(f"{e.FILE_PATH}")

    e.log(f"{e.apply_hook('print_cuda')}=")
    e.apply_hook("create_file", path=e.FILE_PATH)
    e.log("LOG: Done")

experiment.run_if_main()