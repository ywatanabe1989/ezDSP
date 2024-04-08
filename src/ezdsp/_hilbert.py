#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-08 13:58:36 (ywatanabe)"

import ezdsp
from ezdsp.nn import Hilbert
from torch_fn import torch_fn


@torch_fn
def hilbert(
    x,
    dim=-1,
):
    y = Hilbert(dim=dim)(x)
    return y[..., 0], y[..., 1]


def _get_scipy_x(t_sec, fs):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import chirp, hilbert

    # duration = 1.0
    # fs = 400.0

    duration = t_sec
    samples = int(fs * duration)
    t = np.arange(samples) / fs

    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= 1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)

    x = signal

    x = ezdsp.ensure_3d(x)

    t = torch.linspace(0, T_SEC, x.shape[-1])

    return x, t, fs


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.signal import chirp

    T_SEC = 1.0  # 0
    FS = 400  # 128

    # xx, tt, fs = _get_scipy_x(T_SEC, FS)
    xx, tt, fs = ezdsp.demo_sig(t_sec=T_SEC, fs=FS, sig_type="chirp")

    pha, amp = hilbert(
        xx,
        dim=-1,
    )  # (32, 19, 1280, 2)

    fig, axes = plt.subplots(nrows=2)
    axes[0].plot(tt, xx[0, 0], label="orig")
    axes[0].plot(tt, amp[0, 0], label="amp")
    axes[1].plot(tt, pha[0, 0], label="phase")
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    plt.show()
