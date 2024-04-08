#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-09 01:02:23 (ywatanabe)"

import matplotlib

matplotlib.use("Agg")
import ezdsp
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from numpy_fn import numpy_fn


# Functions
def calc_norm_resample_filt_hilbert(xx, tt, fs, sig_type, verbose=True):
    sigs = {"index": ("signal", "time", "fs")}  # Collector

    sigs[f"orig"] = (xx, tt, fs)

    # Normalization
    sigs["z_normed"] = (ezdsp.norm.z(xx), tt, fs)
    sigs["minmax_normed"] = (ezdsp.norm.minmax(xx), tt, fs)

    # Resampling
    sigs["resampled"] = (
        ezdsp.resample(xx, fs, TGT_FS),
        tt[:: int(fs / TGT_FS)],
        TGT_FS,
    )

    # Noise injection
    sigs["gaussian_noise_added"] = (ezdsp.add_noise.gauss(xx), tt, fs)
    sigs["white_noise_added"] = (ezdsp.add_noise.white(xx), tt, fs)
    sigs["pink_noise_added"] = (ezdsp.add_noise.pink(xx), tt, fs)
    sigs["brown_noise_added"] = (ezdsp.add_noise.brown(xx), tt, fs)

    # Filtering
    _xx = ezdsp.filt.bandpass(xx, fs, low_hz=LOW_HZ, high_hz=HIGH_HZ)
    _tt = ezdsp.filt.truncate_time(tt, _xx.shape[-1])
    sigs[f"bandpass_filted ({LOW_HZ} - {HIGH_HZ} Hz)"] = (_xx, _tt, fs)

    _xx = ezdsp.filt.bandstop(xx, fs, low_hz=LOW_HZ, high_hz=HIGH_HZ)
    _tt = ezdsp.filt.truncate_time(tt, _xx.shape[-1])
    sigs[f"bandstop_filted ({LOW_HZ} - {HIGH_HZ} Hz)"] = (_xx, _tt, fs)

    _xx = ezdsp.filt.gauss(xx, sigma=SIGMA)
    _tt = ezdsp.filt.truncate_time(tt, _xx.shape[-1])
    sigs[f"bandstop_gauss (sigma = {SIGMA})"] = (_xx, _tt, fs)

    # Hilbert Transformation
    pha, amp = ezdsp.hilbert(xx)
    sigs["hilbert_amp"] = (amp, tt, fs)
    sigs["hilbert_pha"] = (pha, tt, fs)

    sigs = pd.DataFrame(sigs).set_index("index")

    if verbose:
        print(sigs.index)
        print(sigs.columns)

    return sigs


def plot_signals(plt, sigs, sig_type):
    fig, axes = plt.subplots(nrows=len(sigs.columns), sharex=True)

    i_batch = 0
    i_ch = 0
    for ax, (i_col, col) in zip(axes, enumerate(sigs.columns)):

        # Main
        xx, tt, fs = sigs[col]

        try:
            ax.plot(
                tt,
                xx[i_batch, i_ch],
                label=col,
                c=CC["red"] if col == "hilbert_amp" else CC["blue"],
            )

            if col == "hilbert_amp":  # add the original signal to the ax
                _col = "orig"
                (
                    _xx,
                    _tt,
                    _fs,
                ) = sigs[_col]
                ax.plot(_tt, _xx[i_batch, i_ch], label=_col, c=CC["blue"])

        except Exception as e:
            print(e)

        # Adjustments
        ax.legend(loc="upper left")
        ax.set_xlim(tt[0], tt[-1])
        ax = mngs.plt.ax.set_n_ticks(ax)

    fig.supxlabel("Time [s]")
    fig.supylabel("Voltage")
    fig.suptitle(sig_type)
    fig.tight_layout()
    return fig


def plot_wavelet(plt, sigs, sig_col, sig_type):

    xx, tt, fs = sigs[sig_col]

    # Wavelet Transformation
    wavelet_coef, ff_ww = ezdsp.wavelet(xx, fs)

    i_batch = 0
    i_ch = 0

    # Main
    fig, axes = plt.subplots(nrows=2, sharex=True)
    # Signal
    axes[0].plot(
        tt,
        xx[i_batch, i_ch],
        label=sig_col,
        c=CC["blue"],
    )

    # Adjusts
    axes[0].legend(loc="upper left")
    axes[0].set_xlim(tt[0], tt[-1])
    axes[0].set_ylabel("Voltage")
    # axes[0] = mngs.plt.ax.set_n_ticks(axes[0])

    # Wavelet Spectrogram
    axes[1].imshow(
        wavelet_coef[i_batch, i_ch],
        aspect="auto",
        extent=[tt[0], tt[-1], 512, 1],
        label="wavelet_coefficient",
    )
    # axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    # axes[1].legend(loc="upper left")
    axes[1].invert_yaxis()

    fig.supxlabel("Time [s]")
    fig.suptitle(sig_type)
    fig.tight_layout()
    return fig


def plot_psd(plt, sigs, sig_col, sig_type):

    xx, tt, fs = sigs[sig_col]

    # Power Spetrum Density
    psd, ff_pp = ezdsp.psd(xx, fs)

    # Plots
    i_batch = 0
    i_ch = 0
    fig, axes = plt.subplots(nrows=2, sharex=False)

    # Signal
    axes[0].plot(
        tt,
        xx[i_batch, i_ch],
        label=sig_col,
        c=CC["blue"],
    )
    # Adjustments
    axes[0].legend(loc="upper left")
    axes[0].set_xlim(tt[0], tt[-1])
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Voltage")
    # axes[0] = mngs.plt.ax.set_n_ticks(axes[0])

    # PSD
    axes[1].plot(ff_pp, psd[i_batch, i_ch], label="PSD")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Power [uV^2 / Hz]")
    axes[1].set_xlabel("Frequency [Hz]")

    fig.suptitle(sig_type)
    fig.tight_layout()
    return fig


def plot_pac(plt, sigs, sig_col, sig_type):
    x_3d = sigs[sig_col].signal
    assert x_3d.ndim == 3
    # if sig_type == tensorpac -> (batch_size, n_segments, seq_len)
    # if sig_type != tensorpac -> (batch_size, n_chs, seq_len)

    # To reduce the VRAM load, slice the array into smaller pieces while keeping
    # the n_segments dimension for the tensorpac demo signal.

    # Slices the batch_size to 1
    x_3d = x_3d[:1, ...]
    x_4d = x_3d[np.newaxis, ...]

    print(x_4d.shape)

    # ezdsp.pac recognize the x_4d as (batch_size, n_chs, n_segments, seq_len)
    try:
        pac, freqs_pha, freqs_amp = ezdsp.pac(x_3d, fs)
    except Exception as e:
        print(e)
        pac, freqs_pha, freqs_amp = ezdsp.pac(x_3d, fs, device="cpu")
    fig, ax = mngs.plt.subplots()
    i_batch = 0
    i_ch = 0
    ax.imshow2d(pac[i_batch, i_ch])
    ax = mngs.plt.ax.set_ticks(
        ax,
        xticks=freqs_pha.mean(axis=-1).astype(int),
        yticks=freqs_amp.mean(axis=-1).astype(int),
    )
    ax = mngs.plt.ax.set_n_ticks(ax)
    ax.set_xlabel("Frequency for phase [Hz]")
    ax.set_ylabel("Frequency for amplitude [Hz]")
    ax.set_title("PAC")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import os

    plt, CC = mngs.plt.configure_mpl(plt, line_width=0.2, verbose=True)

    # Parameters
    T_SEC = 4
    SIG_TYPES = [
        "uniform",
        "gauss",
        "periodic",
        "chirp",
        "ripple",
        "meg",
        "tensorpac",
    ]
    SRC_FS = 512
    TGT_FS = 256
    FREQS_HZ = [10, 30, 100]
    LOW_HZ = 20
    HIGH_HZ = 50
    SIGMA = 10

    sdir = "./example_outputs/"
    os.makedirs(sdir, exist_ok=True)

    for i_sig_type, sig_type in enumerate(SIG_TYPES):
        # Demo Signal
        xx, tt, fs = ezdsp.demo_sig(
            t_sec=T_SEC, fs=SRC_FS, freqs_hz=FREQS_HZ, sig_type=sig_type
        )
        xx_orig = xx.copy()

        # for consistency among various demo signals
        if sig_type == "tensorpac":
            # (batch_size, n_chs, n_segments, seq_len) -> (batch_size, n_segments, seq_len)
            xx = xx_orig[:, 0]
        else:
            xx = xx_orig

        # Apply calculations on the original signal
        sigs = calc_norm_resample_filt_hilbert(xx, tt, fs, sig_type)

        # Plots signals
        fig = plot_signals(plt, sigs, sig_type)
        mngs.io.save(fig, sdir + f"{i_sig_type}_{sig_type}/1_signals.png")

        # Plots wavelet coefficients and PSD
        for sig_col in sigs.columns:

            if "hilbert" in sig_col:
                continue

            fig = plot_wavelet(plt, sigs, sig_col, sig_type)
            mngs.io.save(
                fig, sdir + f"{i_sig_type}_{sig_type}/2_wavelet_{sig_col}.png"
            )

            fig = plot_psd(plt, sigs, sig_col, sig_type)
            mngs.io.save(
                fig, sdir + f"{i_sig_type}_{sig_type}/3_psd_{sig_col}.png"
            )

            fig = plot_pac(plt, sigs, sig_col, sig_type)
            mngs.io.save(
                fig, sdir + f"{i_sig_type}_{sig_type}/4_pac_{sig_col}.png"
            )

"""
rm -rf ./example_outputs/
mkdir ./example_outputs/
python ./example.py | tee ./example_outputs/example.py.log
"""
