#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-08 14:19:17 (ywatanabe)"

import math
import warnings

import mngs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_fn import torch_fn


class BaseFilter1D(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.kernel = None

    @property
    def kernel_size(
        self,
    ):
        return _to_even(self.kernel.shape[-1])

    @property
    def radius(
        self,
    ):
        return self.kernel_size // 2

    def forward(self, x, t=None):
        """Apply the filter to input signal x with shape: (batch_size, n_chs, seq_len)"""

        x = ezdsp.ensure_3d(x)
        seq_len = x.shape[-1]

        # Ensure the kernel is initialized
        if self.kernel is None:
            self.init_kernel()
            if self.kernel is None:
                raise ValueError("Filter kernel has not been initialized.")

        # Edge handling and convolution
        extension_length = self.radius
        first_segment = x[:, :, :extension_length].flip(dims=[-1])
        last_segment = x[:, :, -extension_length:].flip(dims=[-1])
        extended_x = torch.cat([first_segment, x, last_segment], dim=-1)

        channels = extended_x.size(1)

        kernel = (
            self.kernel.expand(channels, 1, -1)
            .to(extended_x.device)
            .to(extended_x.dtype)
        )

        x_filted_extended = F.conv1d(
            extended_x, kernel, padding=0, groups=channels
        )[..., :seq_len]

        assert x.shape == x_filted_extended.shape

        # # Remove edges
        # x_filted_extended[..., : self.radius] *= 0
        # x_filted_extended[..., -self.radius :] *= 0

        nn_remove = x_filted_extended.shape[-1] // 4
        x_filted_extended = x_filted_extended[..., nn_remove:-nn_remove]
        return x_filted_extended


class GaussianFilter(BaseFilter1D):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = _to_even(sigma)
        self.init_kernel()

    def init_kernel(self):
        # Create a Gaussian kernel
        kernel_size = self.sigma * 6  # +/- 3SD
        kernel_range = torch.arange(kernel_size) - kernel_size // 2
        kernel = torch.exp(-0.5 * (kernel_range / self.sigma) ** 2)
        kernel /= kernel.sum()  # Normalize the kernel
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)


class LowPassFilter(BaseFilter1D):
    def __init__(self, cutoff_hz, fs, kernel_size=None):
        super().__init__()
        self.cutoff_hz = cutoff_hz
        self.fs = fs

        kernel_size = (
            _to_even(int(1 / cutoff_hz * fs * 3))
            if kernel_size is None
            else _to_even(kernel_size)
        )

        self.init_kernel(kernel_size)

    def init_kernel(self, kernel_size):
        freqs = torch.fft.fftfreq(kernel_size, d=1 / self.fs)
        kernel = torch.zeros(kernel_size)
        kernel[
            freqs <= self.cutoff_hz
        ] = 1  # Allow frequencies below the cutoff
        kernel = torch.fft.ifft(kernel).real
        # kernel /= kernel.abs().sum()  # Normalize the kernel by its absolute sum
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)


class HighPassFilter(BaseFilter1D):
    def __init__(self, cutoff_hz, fs, kernel_size=None):
        super().__init__()
        self.cutoff_hz = cutoff_hz
        self.fs = fs
        kernel_size = (
            _to_even(int(1 / cutoff_hz * fs * 3))
            if kernel_size is None
            else _to_even(kernel_size)
        )
        self.init_kernel(kernel_size)

    def init_kernel(self, kernel_size):
        freqs = torch.fft.fftfreq(kernel_size, d=1 / self.fs)
        kernel = torch.zeros(kernel_size)
        kernel[
            freqs >= self.cutoff_hz
        ] = 1  # Allow frequencies above the cutoff
        kernel = torch.fft.ifft(kernel).real
        # kernel /= kernel.abs().sum()  # Normalize the kernel by its absolute sum
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)


class BandPassFilter(BaseFilter1D):
    def __init__(self, low_hz, high_hz, fs, kernel_size=None):
        super().__init__()

        assert 0 < low_hz
        assert low_hz < high_hz
        assert high_hz <= fs / 2

        kernel_size = (
            _to_even(int(1 / low_hz * fs * 3))
            # _to_even(int(low_hz * fs * 3))
            if kernel_size is None
            else _to_even(kernel_size)
        )

        self.low_hz = low_hz
        self.high_hz = high_hz
        self.fs = fs
        self.init_kernel(kernel_size)

    def init_kernel(self, kernel_size):
        freqs = torch.fft.fftfreq(kernel_size, d=1 / self.fs)
        kernel = torch.zeros(kernel_size)
        kernel[(self.low_hz <= freqs) & (freqs <= self.high_hz)] = 1
        kernel = torch.fft.ifft(kernel).real
        # kernel /= kernel.sum()
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)


class BandStopFilter(BaseFilter1D):
    def __init__(self, low_hz, high_hz, fs, kernel_size=None):
        super().__init__()
        kernel_size = (
            _to_even(int(1 / low_hz * fs * 3))
            # _to_even(int(low_hz * fs * 3))
            if kernel_size is None
            else _to_even(kernel_size)
        )
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.fs = fs
        self.init_kernel(kernel_size)

    def init_kernel(self, kernel_size):
        freqs = torch.fft.fftfreq(kernel_size, d=1 / self.fs)
        kernel = torch.ones(kernel_size)
        kernel[(freqs >= self.low_hz) & (freqs <= self.high_hz)] = 0
        kernel = torch.fft.ifft(
            kernel
        ).real  # Inverse FFT to get the time-domain kernel
        # kernel /= kernel.sum()  # Normalize the kernel
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)


def _to_even(n):
    if n % 2 == 0:
        return n
    else:
        return n - 1
