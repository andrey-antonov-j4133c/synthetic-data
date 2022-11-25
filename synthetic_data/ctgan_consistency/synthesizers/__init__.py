"""Synthesizers module."""

from synthetic_data.ctgan_consistency.synthesizers.ctgan import CTGAN
from synthetic_data.ctgan_consistency.synthesizers.tvae import TVAE

__all__ = (
    'CTGAN',
    'TVAE'
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
