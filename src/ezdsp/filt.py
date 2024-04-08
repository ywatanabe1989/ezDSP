#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-08 13:50:22 (ywatanabe)"


import ezdsp
from ezdsp.nn import BandPassFilter, BandStopFilter, GaussianFilter
from torch_fn import torch_fn


@torch_fn
def gauss(x, sigma):
    return GaussianFilter(sigma)(x)


@torch_fn
def bandpass(x, samp_rate, low_hz, high_hz):
    return BandPassFilter(low_hz, high_hz, samp_rate)(x)


@torch_fn
def bandstop(x, samp_rate, low_hz, high_hz):
    return BandStopFilter(low_hz, high_hz, samp_rate)(x)


@torch_fn
def truncate_time(tt, seq_len):
    seq_diff = len(tt) - seq_len
    if seq_diff > 0:
        return tt[seq_diff // 2 :][:seq_len]
    else:
        return tt


if __name__ == "__main__":
    import torch

    t_sec = 10
    src_fs = 1024
    tgt_fs = 128
    freqs_hz = [30, 60, 100, 200, 1000]

    xx, tt, fs = ezdsp.demo_sig(
        t_sec=t_sec, fs=src_fs, freqs_hz=freqs_hz, sig_type="ripple"
    )
    resampled = ezdsp.resample(xx, src_fs, tgt_fs)
    tt_resampled = ezdsp.resample(tt, src_fs, tgt_fs)

    # Filtering
    filted_bp = ezdsp.filt.bandpass(xx, fs, low_hz=20, high_hz=50)
    filted_bs = ezdsp.filt.bandstop(xx, fs, low_hz=20, high_hz=50)
    filted_gauss = ezdsp.filt.gauss(xx, sigma=3)

    # Plots
    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=True)
    i_batch = 0
    i_ch = 0
    axes[0].plot(tt, xx[i_batch, i_ch], label="Original")
    axes[1].plot(tt_resampled, resampled[i_batch, i_ch], label="Resampled")
    axes[2].plot(
        truncate_time(tt, filted_bp.shape[-1]),
        filted_bp[i_batch, i_ch],
        label="Bandpass-filtered",
    )
    axes[3].plot(
        truncate_time(tt, filted_bs.shape[-1]),
        filted_bs[i_batch, i_ch],
        label="Bandstop-filtered",
    )
    axes[4].plot(
        truncate_time(tt, filted_gauss.shape[-1]),
        filted_gauss[i_batch, i_ch],
        label="Gaussian-filtered",
    )
    for ax in axes:
        ax.legend(loc="upper left")
    plt.show()
