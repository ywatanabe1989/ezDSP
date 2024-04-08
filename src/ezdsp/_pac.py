#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-09 00:49:47 (ywatanabe)"

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
    import ezdsp

    # Parameters
    FS = 1024
    T_SEC = 4

    xx, tt, fs = ezdsp.demo_sig(
        batch_size=1, n_chs=1, fs=FS, t_sec=T_SEC, sig_type="tensorpac"
    )
    pac, pha_bands, amp_bands = ezdsp.pac(
        xx, fs, pha_n_bands=30, amp_n_bands=50
    )

    # xx, tt, fs = ezdsp.demo_sig(
    #     batch_size=1, n_chs=1, fs=FS, t_sec=T_SEC, sig_type="ripple"
    # )

    # Plots the calculated PAC
    i_batch = 0
    i_ch = 0
    fig, ax = mngs.plt.subplots()
    ax.imshow2d(pac[i_batch, i_ch])
    ax = mngs.plt.ax.set_ticks(
        ax,
        xticks=np.array(pha_bands).mean(axis=-1).astype(int),
        yticks=np.array(amp_bands).mean(axis=-1).astype(int),
    )
    ax = mngs.plt.ax.set_n_ticks(ax)
    ax.set_xlabel("Frequency for phase [Hz]")
    ax.set_ylabel("Frequency for amplitude [Hz]")
    ax.set_title("PAC values")
    plt.show()
