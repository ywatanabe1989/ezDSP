#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-05 21:11:16 (ywatanabe)"

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

# class ModulationIndex(nn.Module):
#     def __init__(self, n_bins=18):
#         super(ModulationIndex, self).__init__()
#         self.n_bins = n_bins
#         # Create bin cutoffs tensor
#         self.register_buffer('pha_bin_cutoffs', torch.linspace(-np.pi, np.pi, n_bins + 1))

#     def forward(self, pha, amp):
#         """
#         pha.shape: (batch_size, n_chs, seq_len)
#         amp.shape: (batch_size, n_chs, seq_len)
#         """
#         # Digitize phase data to find which bin each phase value belongs to
#         # bins = torch.digitize(pha, self.pha_bin_cutoffs) - 1  # Subtract 1 for 0-based indexing
#         bins = torch.bucketize(pha, self.pha_bin_cutoffs) - 1  # Subtract 1 for 0-based indexing

#         # We need to ensure we do not have bins outside our expected range due to precision errors
#         bins = torch.clamp(bins, 0, self.n_bins - 1)

#         # Initialize a tensor to hold summed amplitudes and counts for normalization
#         amp_sum = torch.zeros(*pha.shape[:-1], self.n_bins, device=pha.device)
#         counts = torch.zeros(*pha.shape[:-1], self.n_bins, device=pha.device)

#         # Accumulate amp sums and counts for each bin
#         for i_bin in range(self.n_bins):
#             mask = bins == i_bin
#             amp_sum[..., i_bin] = (amp * mask).sum(dim=-1)
#             counts[..., i_bin] = mask.sum(dim=-1)

#         # Compute mean amplitude per bin avoiding division by zero
#         amp_means = amp_sum / counts.clamp(min=1)

#         # Convert amplitude means to probabilities
#         amp_probs = amp_means / amp_means.sum(dim=-1, keepdim=True)

#         # Calculate Modulation Index (MI)
#         n = torch.tensor(self.n_bins, dtype=torch.float32, device=pha.device)
#         MI = 1 + (1 / n.log()) * (amp_probs * amp_probs.clamp(min=1e-9).log()).sum(dim=-1)

#         return MI


class ModulationIndexLayer(nn.Module):
    def __init__(self, expand_dim=0, n_bins=18):
        super().__init__()
        self.register_buffer("n_bins", torch.tensor(n_bins))
        self.register_buffer(
            "pha_bin_cutoffs", torch.linspace(-np.pi, np.pi, n_bins + 1)
        )
        self.register_buffer("expand_dim", torch.tensor(expand_dim))

    def forward(self, pha, amp):
        amp_means = self.bin_amplitude(pha, amp)  # [..., self.n_bins]
        amp_probs = self.amp_means_to_probs(amp_means)
        MI = self.amp_probs_to_MI(amp_probs)
        return MI

    def bin_amplitude(self, pha, amp):
        """Couples phase and amplitude with the binning method."""
        self.n_bins = self.n_bins.type_as(pha).int()

        if pha.ndim <= 1:  # to 2D
            for _ in range(2 - pha.ndim):
                pha, amp = pha.unsqueeze(0), amp.unsqueeze(0)

        pha, amp = self.expand_on_a_mesh_grid(pha, amp)

        ## Coupling
        amp_means = []
        for i_bin in range(self.n_bins):  # fixme; Do not use for loop.
            low = self.pha_bin_cutoffs[i_bin]
            high = self.pha_bin_cutoffs[i_bin + 1]
            mask = ((low < pha) * (pha < high)).type_as(pha)
            amp_mean_i_bin = (amp * mask).mean(dim=-1)  # coupling
            amp_means.append(amp_mean_i_bin)
        amp_means = torch.stack(amp_means, dim=-1)
        return amp_means

    def amp_means_to_probs(self, amp_means):
        amp_probs = amp_means / amp_means.sum(-1, keepdims=True)
        return amp_probs

    def amp_probs_to_MI(self, amp_probs):
        n = torch.tensor(self.n_bins).type_as(amp_probs)
        MI = 1 + (1 / n.log()) * (amp_probs * amp_probs.log()).sum(-1)
        return MI

    def expand_on_a_mesh_grid(self, pha, amp):
        """Phases and amplitudes are expanded on rectangle mesh.
        This enables to calculate MIs with not "for loops" but matrix multiplication.
        """
        width = pha.shape[self.expand_dim]
        mesh1, mesh2 = self.mk_mesh_grid(width)

        pha = self.dim_i_to_dim_0(pha, self.expand_dim)
        amp = self.dim_i_to_dim_0(amp, self.expand_dim)

        pha, amp = pha[mesh1, ...], amp[mesh2, ...]

        pha = self.dim_0_to_dim_i(pha, self.expand_dim + 1)
        pha = self.dim_0_to_dim_i(pha, self.expand_dim + 1)
        amp = self.dim_0_to_dim_i(amp, self.expand_dim + 1)
        amp = self.dim_0_to_dim_i(amp, self.expand_dim + 1)
        return pha, amp

    def mk_mesh_grid(self, n):
        coord_1, coord_2 = np.meshgrid(np.arange(n), np.arange(n))
        return coord_1, coord_2

    def dim_i_to_dim_0(self, tensor, dim_i):
        for i in range(dim_i, 0, -1):
            tensor = tensor.transpose(i, i - 1)
        return tensor

    def dim_0_to_dim_i(self, tensor, dim_i):
        for i in range(dim_i):
            tensor = tensor.transpose(i, i + 1)
        return tensor


if __name__ == "__main__":
    ################################################################################
    ## Tensorpac
    ################################################################################
    # https://etiennecmb.github.io/tensorpac/auto_examples/pac/plot_pac_methods.html#sphx-glr-auto-examples-pac-plot-pac-methods-py
    import matplotlib.pyplot as plt
    from tensorpac import Pac
    from tensorpac.signals import pac_signals_wavelet

    f_pha = 10  # frequency phase for the coupling
    f_amp = 100  # frequency amplitude for the coupling
    n_epochs = 20  # number of trials
    n_times = 4000  # number of time points
    sf = 512.0  # sampling frequency
    data, time = pac_signals_wavelet(
        sf=sf,
        f_pha=f_pha,
        f_amp=f_amp,
        noise=0.8,
        n_epochs=n_epochs,
        n_times=n_times,
    )

    # define a pac object and extract high-resolution phases and amplitudes using
    # Morlet's wavelets
    p = Pac(f_pha="hres", f_amp="hres", dcomplex="wavelet")
    # etract all of the phases and amplitudes
    phases = p.filter(sf, data, ftype="phase", n_jobs=1)
    amplitudes = p.filter(sf, data, ftype="amplitude", n_jobs=1)

    ## Plot PAC determined with Tensorpac package
    plt.figure(figsize=(14, 8))
    fig, ax = plt.subplots()
    k = 2
    p.idpac = (k, 0, 0)
    xpac = p.fit(phases, amplitudes)
    title = p.method.replace(" (", f" ({k})\n(")
    p.comodulogram(xpac.mean(-1), title=title, cmap="viridis")
    # fig.show()

    print(phases.shape)
    print(amplitudes.shape)
    print(xpac.shape)
    ################################################################################
    ################################################################################

    ## Get the tensorpac's samples (phases and ampiltudes)
    pha_axis, amp_axis = p.xvec, p.yvec
    n_pha, n_amp = len(phases), len(amplitudes)
    phases = torch.FloatTensor(phases)  # shape: [50, 20, 4000]
    amplitudes = torch.FloatTensor(amplitudes)  # shape: [50, 20, 4000]

    ## Convert the sample into 4D array
    # pha_1d, amp_1d = phases[0,0].cuda(), amplitudes[0,0].cuda()
    # pha_2d, amp_2d = phases[0].cuda(), amplitudes[0].cuda()
    # pha_3d, amp_3d = phases.cuda(), amplitudes.cuda()
    pha_4d, amp_4d = phases.unsqueeze(0).cuda(), amplitudes.unsqueeze(0).cuda()

    BS = 4
    zeros_4d = torch.zeros_like(pha_4d).repeat(BS - 1, 1, 1, 1)
    pha_4d = torch.cat([pha_4d, zeros_4d], dim=0)
    amp_4d = torch.cat([amp_4d, zeros_4d], dim=0)

    ## Calculte Modulation Index in PyTorch (, which enables GPU calculation)
    # mi = ModulationIndex(expand_dim=0).cuda()
    # out1 = mi(pha_1d, amp_1d)
    # print(out1)

    # mi = ModulationIndex(expand_dim=0).cuda()
    # out2 = mi(pha_2d, amp_2d)
    # print(out2)

    # mi = ModulationIndex(expand_dim=0).cuda()
    # out3 = mi(pha_3d, amp_3d)
    # print(out3)

    mi = ModulationIndexLayer(n_bins=18).cuda()  # Initialization
    # mi = ModulationIndexLayer(expand_dim=1, n_bins=18).cuda()  # Initialization
    out4 = mi(pha_4d, amp_4d)
    print(out4)

    ## Convert y-axis
    xpac_torch = out4[0]
    pac_torch = xpac_torch.mean(-1)
    indi_y_converted = np.arange(len(pac_torch))[::-1].copy()
    pac_torch = pac_torch[torch.tensor(indi_y_converted)]
    amp_axis = amp_axis[indi_y_converted]

    ## Plot PAC determined with our implementation
    pha_axis_dict = {i: np.round(pha_axis[i], 0) for i in range(len(pha_axis))}
    amp_axis_dict = {i: np.round(amp_axis[i], 0) for i in range(len(amp_axis))}

    fig, ax = plt.subplots()
    sns.heatmap(pac_torch.cpu().numpy(), cmap="viridis")
    title = 'Pytorch calculation on GPU without any "for loops"'
    ax.set_title(title)

    ax.set_xticks(np.arange(pac_torch.shape[0]))
    ax.set_xticklabels([pha_axis_dict[i] for i in range(pac_torch.shape[0])])

    ax.set_yticks(np.arange(pac_torch.shape[1]))
    ax.set_yticklabels([amp_axis_dict[i] for i in range(pac_torch.shape[1])])

    fig.show()
