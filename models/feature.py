from typing import final
import torch
import torch.nn as nn

from .base_models import load_backbone


@final
class Model(nn.Module):

    def __init__(self, dataset, args):
        super(Model, self).__init__()

        # submodules
        self.backbone, self.feat_dim = load_backbone(args.backbone_type, args.backbone_weight)
        if self.feat_dim is None:
            self.feat_dim = dataset.feat_dim


    def forward(self, batch, require_loss=True):
        assert self.backbone is not None

        feature = self.backbone(batch["image"], batch["main_bbox"])
        return feature