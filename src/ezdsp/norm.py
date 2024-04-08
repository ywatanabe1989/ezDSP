#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-08 14:06:38 (ywatanabe)"

from torch_fn import torch_fn as _torch_fn


@_torch_fn
def z(x, dim=-1):
    return (x - x.mean(dim=dim, keepdim=True)) / x.std(dim=dim, keepdim=True)


@_torch_fn
def minmax(x, amp=1.0, dim=-1, fn="mean"):
    MM = x.max(dim=dim, keepdims=True)[0].abs()
    mm = x.min(dim=dim, keepdims=True)[0].abs()
    return amp * x / _torch.maximum(MM, mm)
