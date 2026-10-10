import pytest
import torch

from elmur_klpo.exact_enumeration import enumerate_trajectories
from elmur_klpo.toy_env import ToyEnvironment


@pytest.fixture(scope="session", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="session")
def env():
    return ToyEnvironment()


@pytest.fixture(scope="session")
def batch(env):
    return enumerate_trajectories(env)
