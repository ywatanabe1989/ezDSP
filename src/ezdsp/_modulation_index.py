#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-09 02:03:15 (ywatanabe)"

import torch
from ezdsp.nn import ModulationIndex
from torch_fn import torch_fn


@torch_fn
def modulation_index(pha, amp, n_bins=18, device="cuda"):
    """
    pha: (batch_size, n_chs, n_freqs_pha, n_segments, seq_len)
    amp: (batch_size, n_chs, n_freqs_amp, n_segments, seq_len)
    """
    return ModulationIndex(n_bins=n_bins)(pha, amp)


def calc_pac_with_tensorpac(xx, fs, t_sec):
    import tensorpac

    # Morlet's Wavelet Transfrmation
    p = tensorpac.Pac(f_pha="hres", f_amp="demon", dcomplex="wavelet")
    #
    # n_bands: | lres, 10 | mres, 30 | hres, 50 | demon, 70 | hulk, 100 |

    # Bandpass Filtering and Hilbert Transformation
    i_batch, i_ch = 0, 0
    phases = p.filter(
        fs, xx[i_batch, i_ch], ftype="phase", n_jobs=1
    )  # (50, 20, 2048)
    amplitudes = p.filter(
        fs, xx[i_batch, i_ch], ftype="amplitude", n_jobs=1
    )  # (50, 20, 2048)

    # Calculates xpac
    k = 2
    p.idpac = (k, 0, 0)
    xpac = p.fit(phases, amplitudes)  # (50, 30, 20)
    pac = xpac.mean(axis=-1)  # (50, 30)
    # Transposes `pac_tp` to make the x-axis for phase and the y-axis for amplitude.
    pac = pac.T  # (30, 50)

    freqs_amp = p.f_amp.mean(axis=-1)
    freqs_pha = p.f_pha.mean(axis=-1)

    return phases, amplitudes, freqs_pha, freqs_amp, pac
    # return phases and amplitudes for future use in ezDSP calculation
    # to confirm the calculation of Modulation Index, without considering
    # bandpass filtering and hilbert transformation.


@torch_fn
def reshape(x, batch_size=1, n_chs=1):
    return (
        torch.tensor(x)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, n_chs, 1, 1, 1)
    )


if __name__ == "__main__":
    import ezdsp
    import matplotlib.pyplot as plt
    import mngs

    # Congig
    mngs.plt.configure_mpl(plt, fig_scale=5)

    # Parameters
    FS = 512
    T_SEC = 8

    # Demo signal
    xx, tt, fs = ezdsp.demo_sig(
        fs=FS, t_sec=T_SEC, sig_type="tensorpac"
    )  # (8, 19, 20, 512)

    # Tensorpac calculation
    (
        pha_tp,
        amp_tp,
        freqs_pha_tp,  # (10,)
        freqs_amp_tp,  # (30,)
        pac_tp,  # (30, 10)
    ) = calc_pac_with_tensorpac(
        xx,
        fs,
        t_sec=T_SEC,
    )

    # ezDSP calculation (on CPU now, due to the limitation in computational resources)
    i_batch, i_ch = 0, 0
    import ipdb

    ipdb.set_trace()
    pac_ed = ezdsp.modulation_index(
        reshape(pha_tp), reshape(amp_tp), device="cpu"
    )[
        i_batch, i_ch
    ]  # (10, 30)

    # Plots
    fig, axes = mngs.plt.subplots(ncols=3, sharex=True, sharey=True)

    # To align scalebars
    vmin = min(np.min(pac_ed), np.min(pac_tp), np.min(pac_ed - pac_tp))
    vmax = max(np.max(pac_ed), np.max(pac_tp), np.max(pac_ed - pac_tp))

    # EZDSP
    axes[0].imshow2d(
        pac_ed,
        cbar_label="PAC values",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("ezDSP (on GPU)")

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
        "./example_outputs/modulation_index_calculation_with_ezDSP_and_Tensorpac.png",
    )
