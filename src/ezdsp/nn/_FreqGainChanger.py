#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-08 14:38:01 (ywatanabe)"

import julius
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class FreqGainChanger(nn.Module):
    def __init__(self, n_bands, samp_rate, dropout_ratio=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.n_bands = n_bands
        self.samp_rate = samp_rate
        # self.register_buffer("ones", torch.ones(self.n_bands))

    def forward(self, x):
        """x: [batch_size, n_chs, seq_len]"""
        if self.training:
            x = julius.bands.split_bands(
                x, self.samp_rate, n_bands=self.n_bands
            )
            freq_gains = (
                torch.rand(self.n_bands)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .to(x.device)
                + 0.5
            )
            freq_gains = F.softmax(freq_gains, dim=0)
            x = (x * freq_gains).sum(axis=0)

        return x


if __name__ == "__main__":
    # Parameters
    N_BANDS = 10
    SAMP_RATE = 1000
    BS, N_CHS, SEQ_LEN = 16, 360, 1000

    # Demo data
    x = torch.rand(BS, N_CHS, SEQ_LEN).cuda()

    # Feedforward
    fgc = FreqGainChanger(N_BANDS, SAMP_RATE).cuda()
    # fd.eval()
    y = fgc(x)
    y.sum().backward()
