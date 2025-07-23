# Copyright (c) OpenMMLab. All rights reserved.
from .rtmcc_head import RTMCCHead
from .rtmw_head import RTMWHead
from .simcc_head import SimCCHead
from .cig_head import CIGHead

__all__ = ['SimCCHead', 'RTMCCHead', 'RTMWHead', 'CIGHead']
