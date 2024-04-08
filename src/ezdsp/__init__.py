#!/usr/bin/env python3

from . import PARAMS, add_noise, filt, norm, ref
from ._demo_sig import demo_sig
from ._hilbert import hilbert
from ._misc import ensure_3d
from ._mne import get_eeg_pos
from ._modulation_index import modulation_index
from ._pac import pac
from ._psd import psd
from ._resample import resample
from ._transform import to_segments, to_sktime_df
from ._wavelet import wavelet

__copyright__ = "Copyright (C) 2024 Yusuke Watanabe"
__version__ = "1.0.0"
__license__ = "MIT"
__author__ = "ywatanabe1989"
__author_email__ = "ywata1989@gmail.com"
__url__ = "https://github.com/ywatanabe1989/ezdsp"
