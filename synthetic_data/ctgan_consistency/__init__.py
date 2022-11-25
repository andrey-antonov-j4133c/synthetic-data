# -*- coding: utf-8 -*-

"""Top-level package for ctgan."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.6.1.dev0'

from synthetic_data.ctgan_consistency.demo import load_demo
from synthetic_data.ctgan_consistency.synthesizers.ctgan import CTGAN
from synthetic_data.ctgan_consistency.synthesizers.tvae import TVAE

__all__ = (
    'CTGAN',
    'TVAE',
    'load_demo'
)
