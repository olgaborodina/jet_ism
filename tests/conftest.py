import numpy as np
import pytest


class MockSnap:
    """Minimal dict-like object that mimics h5py snapshot field access.

    Supports snap_data['PartType0/Field'][:] as used throughout jet_ism.
    """

    def __init__(self, internal_energy, electron_abundance):
        self._data = {
            "PartType0/InternalEnergy": np.atleast_1d(
                np.asarray(internal_energy, dtype=np.float64)
            ),
            "PartType0/ElectronAbundance": np.atleast_1d(
                np.asarray(electron_abundance, dtype=np.float64)
            ),
        }

    def __getitem__(self, key):
        return self._data[key]


@pytest.fixture
def make_snap():
    """Factory fixture: make_snap(internal_energy, electron_abundance) -> MockSnap."""
    return MockSnap
