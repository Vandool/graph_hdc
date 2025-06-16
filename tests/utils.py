import logging
import pathlib
import sys

from decouple import config

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = PATH / "assets"
ARTIFACTS_PATH = PATH / "artifacts"
DATASET_TEST_PATH = ARTIFACTS_PATH / "datasets"

LOG_TESTING = config("LOG_TESTING", cast=bool, default=True)
LOG = logging.getLogger("Testing")
LOG.setLevel(logging.DEBUG)
LOG.addHandler(logging.NullHandler())
if LOG_TESTING:
    LOG.addHandler(logging.StreamHandler(sys.stdout))
