#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-08 18:35:30 (ywatanabe)"

import matplotlib

matplotlib.use("Agg")
import ezdsp
import matplotlib.pyplot as plt
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

        # ax = mngs.plt.ax.set_n_ticks(ax)

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

    # Main
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


def save_fig(fig, spath):
    import os

    _sdir = os.path.split(spath)[0]
    os.makedirs(_sdir, exist_ok=True)
    fig.savefig(spath)

    print(f"\nSaved to: {spath}")


def imshow2d(
    ax,
    arr_2d,
    id=None,
    track=True,
    cbar_label=None,
    cmap="viridis",
):
    assert arr_2d.ndim == 2
    # Call the original ax.imshow() method
    arr_2d = arr_2d.T  # Transpose the array to match imshow's expectations
    im = ax.imshow(arr_2d, cmap=cmap)

    # Create a colorbar
    fig = ax.get_figure()
    cbar = fig.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)

    # Invert y-axis to match typical image orientation
    ax.invert_yaxis()

    return fig


def set_ticks(ax, xticks=None, yticks=None):

    if xticks is not None:
        ax.set_xticks(np.arange(0, len(xticks)))
        ax.set_xticklabels(xticks)

    if yticks is not None:
        ax.set_yticks(np.arange(0, len(yticks)))
        ax.set_yticklabels(yticks)

    return ax


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
    pac, freqs_pha, freqs_amp = ezdsp.pac(x_3d, fs)
    fig, ax = plt.subplots()
    i_batch = 0
    i_ch = 0
    fig = imshow2d(ax, pac[i_batch, i_ch])
    ax = set_ticks(
        ax,
        xticks=freqs_pha.mean(axis=-1).astype(int),
        yticks=freqs_amp.mean(axis=-1).astype(int),
    )
    ax = set_n_ticks(ax)
    ax.set_xlabel("Frequency for phase [Hz]")
    ax.set_ylabel("Frequency for amplitude [Hz]")
    ax.set_title("PAC")
    fig.tight_layout()
    return fig


def set_n_ticks(
    ax,
    n_xticks=4,
    n_yticks=4,
):
    """
    Example:
        ax = set_n_ticks(ax)
    """

    if n_xticks is not None:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(n_xticks))

    if n_yticks is not None:
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(n_yticks))

    # Force the figure to redraw to reflect changes
    ax.figure.canvas.draw()

    return ax


def configure_mpl(
    plt,
    # Fig Size
    fig_size_mm=(160, 100),
    fig_scale=1.0,
    # DPI
    dpi_display=100,
    dpi_save=300,
    # Font Size
    font_size_base=8,
    font_size_title=8,
    font_size_axis_label=8,
    font_size_tick_label=7,
    font_size_legend=6,
    # Hide spines
    hide_top_right_spines=True,
    # line
    line_width=0.1,
    # Color transparency
    alpha=1.0,
    # Whether to print configurations or not
    verbose=False,
    **kwargs,
):
    """
    Configures Matplotlib and Seaborn settings for publication-quality plots.
    For axis control, refer to the mngs.plt.ax module.

    Example:
        plt, cc = configure_mpl(plt)

        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        for i_cc, cc_str in enumerate(cc):
            phase_shift = i_cc * np.pi / len(cc)
            y = np.sin(x + phase_shift)
            ax.plot(x, y, color=cc[cc_str], label=f"{cc_str}")
        ax.legend()
        plt.show()

    Parameters:
        plt (matplotlib.pyplot):
            Matplotlib pyplot module.

        fig_size_mm (tuple of int, optional):
            Figure width and height in millimeters. Defaults to (160, 100).

        fig_scale (float, optional):
            Scaling factor for the figure size. Defaults to 1.0.

        dpi_display (int, optional):
            Display resolution in dots per inch. Defaults to 100.

        dpi_save (int, optional):
            Resolution for saved figures in dots per inch. Defaults to 300.

        font_size_title (int, optional):
            Font size for titles. Defaults to 8.

        font_size_axis_label (int, optional):
            Font size for axis labels. Defaults to 8.

        font_size_tick_label (int, optional):
            Font size for tick labels. Defaults to 7.

        font_size_legend (int, optional):
            Font size for legend text. Defaults to 6.

        hide_top_right_spines (bool, optional):
            If True, hides the top and right spines of the plot. Defaults to True.

        alpha (float, optional):
            Color transparency. Defaults to 0.75.

        verbose (bool, optional):
            If True, prints the configuration settings. Defaults to True.

    Returns:
        dict: A dictionary of the custom colors used in the configuration.
    """

    def rgba_to_hex(rgba):
        return "#{:02x}{:02x}{:02x}{:02x}".format(
            int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3] * 255)
        )

    def normalize_rgba(rgba):
        rgba = list(rgba)
        rgba[0] /= 255
        rgba[1] /= 255
        rgba[2] /= 255
        rgba = tuple(rgba)
        return rgba

    COLORS_RGBA = {
        "blue": (0, 128, 192, alpha),
        "red": (255, 70, 50, alpha),
        "pink": (255, 150, 200, alpha),
        "green": (20, 180, 20, alpha),
        "yellow": (230, 160, 20, alpha),
        "grey": (128, 128, 128, alpha),
        "purple": (200, 50, 255, alpha),
        "lightblue": (20, 200, 200, alpha),
        "brown": (128, 0, 0, alpha),
        "darkblue": (0, 0, 100, alpha),
        "orange": (228, 94, 50, alpha),
        "white": (255, 255, 255, alpha),
        "black": (0, 0, 0, alpha),
    }
    COLORS_HEX = {k: rgba_to_hex(v) for k, v in COLORS_RGBA.items()}
    COLORS_RGBA_NORM = {c: normalize_rgba(v) for c, v in COLORS_RGBA.items()}

    # Normalize figure size from mm to inches
    figsize_inch = (fig_size_mm[0] / 25.4, fig_size_mm[1] / 25.4)

    # Update Matplotlib configuration
    plt.rcParams.update(
        {
            # Resolution
            "figure.dpi": dpi_display,
            "savefig.dpi": dpi_save,
            # Figure Size
            "figure.figsize": figsize_inch,
            # Font Size
            "font.size": font_size_base,
            # Title
            "axes.titlesize": font_size_title,
            # Axis
            "axes.labelsize": font_size_axis_label,
            # Ticks
            "xtick.labelsize": font_size_tick_label,
            "ytick.labelsize": font_size_tick_label,
            # Legend
            "legend.fontsize": font_size_legend,
            # Top and Right Axes
            "axes.spines.top": not hide_top_right_spines,
            "axes.spines.right": not hide_top_right_spines,
            # Custom color cycle
            "axes.prop_cycle": plt.cycler(color=COLORS_RGBA_NORM.values()),
            # Line
            "lines.linewidth": line_width,
        }
    )

    if verbose:
        print("\n" + "-" * 40)
        print("Matplotlib has been configured as follows:\n")
        print(f"Figure DPI (Display): {dpi_display} DPI")
        print(f"Figure DPI (Save): {dpi_save} DPI")
        print(
            f"Figure Size (Not the Axis Size): "
            f"{fig_size_mm[0] * fig_scale:.1f} x "
            f"{fig_size_mm[1] * fig_scale:.1f} mm (width x height)"
        )
        print(f"Font Size (Title): {font_size_title} pt")
        print(f"Font Size (X/Y Label): {font_size_axis_label} pt")
        print(f"Font Size (Tick Label): {font_size_tick_label} pt")
        print(f"Font Size (Legend): {font_size_legend} pt")
        print(f"Hide Top and Right Axes: {hide_top_right_spines}")
        print(f"Custom Colors (RGBA):")
        for color_str, rgba in COLORS_RGBA.items():
            print(f"  {color_str}: {rgba}")
        print("-" * 40)

    return plt, COLORS_RGBA_NORM


if __name__ == "__main__":
    import os

    plt, CC = configure_mpl(plt, line_width=0.2, verbose=True)

    # Parameters
    T_SEC = 4
    SIG_TYPES = [
        "uniform",
        # "gauss",
        # "periodic",
        # "chirp",
        # "ripple",
        # "meg",
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
        save_fig(fig, sdir + f"{i_sig_type}_{sig_type}/1_signals.png")

        # Plots wavelet coefficients and PSD
        for sig_col in sigs.columns:

            if "hilbert" in sig_col:
                continue

            fig = plot_wavelet(plt, sigs, sig_col, sig_type)
            save_fig(
                fig, sdir + f"{i_sig_type}_{sig_type}/2_wavelet_{sig_col}.png"
            )

            fig = plot_psd(plt, sigs, sig_col, sig_type)
            save_fig(
                fig, sdir + f"{i_sig_type}_{sig_type}/3_psd_{sig_col}.png"
            )

            fig = plot_pac(plt, sigs, sig_col, sig_type)
            save_fig(
                fig, sdir + f"{i_sig_type}_{sig_type}/4_pac_{sig_col}.png"
            )

"""
rm -rf ./example_outputs/
mkdir ./example_outputs/
python ./example.py | tee ./example_outputs/example.py.log
"""
