#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-08 14:07:29 (ywatanabe)"


import torchaudio.transforms as T
from torch_fn import torch_fn


@torch_fn
def resample(x, src_fs, tgt_fs):
    resampler = T.Resample(src_fs, tgt_fs, dtype=x.dtype).to(x.device)
    return resampler(x)


if __name__ == "__main__":
    # Parameters
    T_SEC = 4
    SIG_TYPE = ["periodic", "chirp", "ripple"][0]
    SRC_FS = 1024
    TGT_FS = 256
    FREQS_HZ = [10, 30, 100]

    # Demo Signal
    xx, tt, fs = ezdsp.demo_sig(
        t_sec=T_SEC, fs=SRC_FS, freqs_hz=FREQS_HZ, type=SIG_TYPE
    )

    # Resampling
    xx_resampled = ezdsp.resample(xx, fs, TGT_FS)
    tt_resampled = ezdsp.resample(tt, fs, TGT_FS)

    # Filtering
    filted_bp = ezdsp.filt.bandpass(xx, fs, low_hz=20, high_hz=50)
    filted_bs = ezdsp.filt.bandstop(xx, fs, low_hz=20, high_hz=50)
    filted_gauss = ezdsp.filt.gauss(xx, sigma=3)

    # Power Spetrum Density
    psd, ff_pp = ezdsp.psd(xx, fs)

    # Wavelet Transformation
    wvlt_coef, ff_ww = ezdsp.wavelet(xx, fs)

    # Hilbert Transformation
    pha, amp = ezdsp.hilbert(xx)

    # Plots
    i_batch = 0
    i_ch = 0
    fig, axes = plt.subplots(nrows=5, sharex=True, sharey=True)
    axes[0].plot(tt, xx[i_batch, i_ch], label="Original")
    axes[1].plot(tt_resampled, xx_resampled[i_batch, i_ch], label="Resampled")
    axes[2].plot(tt, filted_bp[i_batch, i_ch], label="Bandpass-filtered")
    axes[3].plot(tt, filted_bs[i_batch, i_ch], label="Bandstop-filtered")
    axes[4].plot(tt, filted_gauss[i_batch, i_ch], label="Gaussian-filtered")
    for ax in axes:
        ax.legend()
    plt.show()
