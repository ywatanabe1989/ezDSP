#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-09 01:25:00 (ywatanabe)"

import ezdsp
from ezdsp._modulation_index import calc_pac_with_tensorpac
from ezdsp.nn import PAC
from torch_fn import torch_fn


@torch_fn
def pac(
    x,
    fs,
    pha_start_hz=2,
    pha_end_hz=20,
    pha_n_bands=100,
    amp_start_hz=60,
    amp_end_hz=160,
    amp_n_bands=100,
    device="cuda",
):
    """
    Compute the phase-amplitude coupling (PAC) for signals. This function automatically handles inputs as
    PyTorch tensors, NumPy arrays, or pandas DataFrames.

    Arguments:
    - x (torch.Tensor | np.ndarray | pd.DataFrame): Input signal. Shape can be either 4D (batch_size, n_chs, n_segments, seq_len) or 3D (batch_size, n_chs, seq_len), which will be treated as 1 segment.
    - fs (float): Sampling frequency of the input signal.
    - pha_start_hz (float, optional): Start frequency for phase bands. Default is 2 Hz.
    - pha_end_hz (float, optional): End frequency for phase bands. Default is 20 Hz.
    - pha_n_bands (int, optional): Number of phase bands. Default is 100.
    - amp_start_hz (float, optional): Start frequency for amplitude bands. Default is 60 Hz.
    - amp_end_hz (float, optional): End frequency for amplitude bands. Default is 160 Hz.
    - amp_n_bands (int, optional): Number of amplitude bands. Default is 100.

    Returns:
    - torch.Tensor: PAC values. Shape: (batch_size, n_chs, pha_n_bands, amp_n_bands)
    - numpy.ndarray: Phase bands used for the computation.
    - numpy.ndarray: Amplitude bands used for the computation.

    Example:
        FS = 512
        T_SEC = 4
        xx, tt, fs = ezdsp.demo_sig(
            batch_size=1, n_chs=1, fs=FS, t_sec=T_SEC, sig_type="tensorpac"
        )
        pac_values, pha_bands, amp_bands = ezdsp.pac(xx, fs)
    """
    m = PAC(
        fs=fs,
        pha_start_hz=pha_start_hz,
        pha_end_hz=pha_end_hz,
        pha_n_bands=pha_n_bands,
        amp_start_hz=amp_start_hz,
        amp_end_hz=amp_end_hz,
        amp_n_bands=amp_n_bands,
    )
    return m(x), m.BANDS_PHA, m.BANDS_AMP


if __name__ == "__main__":
    import ezdsp as ed

    # Parameters
    FS = 512
    T_SEC = 5

    xx, tt, fs = ezdsp.demo_sig(
        batch_size=1, n_chs=1, fs=FS, t_sec=T_SEC, sig_type="tensorpac"
    )

    # Tensorpac calculation
    (
        pha_tp,
        amp_tp,
        freqs_pha_tp,  # (50,)
        freqs_amp_tp,  # (70,)
        pac_tp,  # (50, 70)
    ) = calc_pac_with_tensorpac(
        xx,
        fs,
        t_sec=T_SEC,
    )

    # ezDSP calculation (on CPU now, due to the limitation in computational resources)
    pac_ed, pha_bands, amp_bands = ed.pac(
        xx, fs, pha_n_bands=50, amp_n_bands=70, device="cpu"
    )
    i_batch, i_ch = 0, 0
    pac_ed = pac_ed[i_batch, i_ch]

    # Plots
    fig, axes = mngs.plt.subplots(ncols=3, sharex=True, sharey=True)

    # To align scalebars
    vmin = min(np.min(pac_ed), np.min(pac_tp))
    vmax = max(np.max(pac_ed), np.max(pac_tp))

    # EZDSP
    axes[0].imshow2d(
        pac_ed,
        cbar_label="PAC values",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("ezDSP")

    # Tensorpac
    axes[1].imshow2d(
        pac_tp,
        cbar_label="PAC values",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Tensorpac")

    # Diff.
    axes[2].imshow2d(
        pac_ed - pac_tp,
        cbar_label="PAC values",
        vmin=vmin,
        vmax=vmax,
    )
    axes[2].set_title("Difference\n(EZDSP - Tensorpac)")

    for ax in axes:
        ax = mngs.plt.ax.set_ticks(
            ax,
            xticks=freqs_pha_tp.astype(int),
            yticks=freqs_amp_tp.astype(int),
        )
        ax = mngs.plt.ax.set_n_ticks(ax)

    fig.supxlabel("Frequency for phase [Hz]")
    fig.supylabel("Frequency for amplitude [Hz]")

    # plt.show()

    mngs.io.save(
        fig,
        "./example_outputs/pac_calculation_with_ezDSP_and_Tensorpac.png",
    )
