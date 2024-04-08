#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-09 03:30:39 (ywatanabe)"

import math
import warnings

import ezdsp
import numpy as np
import scipy
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

        x_filted_extended = self.filtfilt(
            extended_x, kernel, padding="same", groups=channels
        )

        # x_filted_extended = F.conv1d(
        #     extended_x, kernel, padding=0, groups=channels
        # )[..., :seq_len]

        # Ensures the output has the same shape as input
        x_filted = x_filted_extended[..., extension_length:-extension_length]

        assert x.shape == x_filted.shape

        # # Remove edges
        # nn_remove = x_filted.shape[-1] // 8
        # x_filted = x_filted[..., nn_remove:-nn_remove]

        return x_filted

    @staticmethod
    def filtfilt(x, kernel, padding, groups):
        x_filted = F.conv1d(x, kernel, padding=padding, groups=groups)
        x_reversed = torch.flip(x_filted, dims=[-1])
        x_filted_back = F.conv1d(
            x_reversed, kernel, padding=padding, groups=groups
        )
        x_final = torch.flip(x_filted_back, dims=[-1])
        return x_final


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
            _to_even(int(fs * 3))
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


class BandPassFilter(nn.Module):
    # https://raw.githubusercontent.com/mravanelli/SincNet/master/dnn_models.py

    def __init__(self, low_hz, high_hz, fs, order=None):  # kernel_size=None
        # def __init__(self, order=None, low_hz=30, high_hz=60, fs=250, n_chs=19):
        super().__init__()
        self.fs = fs
        nyq = fs / 2

        self.order = (
            self.estimate_fir_order(
                fs,
                low_hz,
                high_hz,
            )
            if order is None
            else order
        )
        self.numtaps = self.order + 1
        filter_npy = scipy.signal.firwin(
            self.numtaps,
            [low_hz, high_hz],
            pass_zero="bandpass",
            fs=fs,
        )

        self.register_buffer(
            "filters", torch.tensor(filter_npy).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        orig_shape = x.shape
        dim = x.ndim
        sig_len_orig = x.shape[-1]

        if dim == 3:
            bs, n_chs, sig_len = x.shape
            x = x.reshape(bs * n_chs, 1, sig_len)

        filted = F.conv1d(
            x, self.filters.type_as(x), padding=int(self.numtaps / 2)
        )
        filted = filted.flip(dims=[-1])  # to backward
        filted = F.conv1d(
            filted, self.filters.type_as(x), padding=int(self.numtaps / 2)
        )
        filted = filted.flip(dims=[-1])  # reverse to the original order

        # filted = filted[..., 1:-1]
        filted = filted[..., :sig_len_orig]

        # please ensure the size is the same with the original input

        if dim == 3:
            filted = filted.reshape(bs, n_chs, -1)

        assert orig_shape == filted.shape

        return filted

    @staticmethod
    def estimate_fir_order(
        fs,
        low_hz,
        high_hz,
        ripple_passband=0.01,
        ripple_stopband=40,
        transition_width=None,
    ):
        if transition_width is None:
            transition_width = low_hz * 0.25

        nyq = fs / 2
        normalized_transition_width = transition_width / nyq

        # Estimate the filter order
        N, beta = scipy.signal.kaiserord(
            ripple_stopband, normalized_transition_width
        )

        N = _to_even(N)

        return N


# class BandPassFilter(BaseFilter1D):
#     def __init__(self, low_hz, high_hz, fs, kernel_size=None):
#         super().__init__()

#         assert 0 < low_hz
#         assert low_hz < high_hz
#         assert high_hz <= fs / 2

#         kernel_size = (
#             _to_even(int(1 / low_hz * fs * 3))
#             # _to_even(int(1 / low_hz * fs * 5))
#             # _to_even(int(fs * 5))
#             if kernel_size is None
#             else _to_even(kernel_size)
#         )

#         self.low_hz = low_hz
#         self.high_hz = high_hz
#         self.fs = fs
#         self.init_kernel(kernel_size)

#     def init_kernel(self, kernel_size):
#         freqs = torch.fft.fftfreq(kernel_size, d=1 / self.fs)
#         kernel = torch.zeros(kernel_size)
#         kernel[(self.low_hz <= freqs) & (freqs <= self.high_hz)] = 1
#         kernel = torch.fft.ifft(kernel).real
#         # kernel /= kernel.sum()
#         self.kernel = kernel.unsqueeze(0).unsqueeze(0)


class BandStopFilter(BaseFilter1D):
    def __init__(self, low_hz, high_hz, fs, kernel_size=None):
        super().__init__()
        kernel_size = (
            # _to_even(int(1 / low_hz * fs * 5))
            # _to_even(int(1 / low_hz * fs * 3))
            _to_even(int(fs * 5))
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
