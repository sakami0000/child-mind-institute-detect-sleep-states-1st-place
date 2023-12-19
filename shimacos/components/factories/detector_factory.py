import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConfig
from mmcv import Config
from mmdet.models import build_detector


class Detector(nn.Module):
    def __init__(self, config: OmegaConfig) -> None:
        detection_config = Config.fromfile(config.detection_config_path)
        self.model = build_detector(
            detection_config.model,
            train_cfg=detection_config.train_cfg,
            test_cfg=detection_config.test_cfg,
        )

