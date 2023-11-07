from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models.detection.faster_rcnn as faster_rcnn
import torchvision.models.detection.backbone_utils as backbone_utils

import json
import numpy as np
import logging


class PositiveBCELoss(nn.Module):
    def __init__(self, class_weight=None):
        super().__init__()
        self.register_buffer("class_weight", class_weight)
        
    def forward(self, logit, target):
        x = F.logsigmoid(logit)*target
        if self.class_weight is not None:
            x = x*self.class_weight
        loss = -(x.mean())
        return loss


def load_backbone(backbone_type, weight_path=None):
    logger = logging.getLogger("load_backbone")
    logger.info(f"Loading {backbone_type} backbone from {weight_path}")

    backbone_type = backbone_type.split("_")
    model_name, weight_type = backbone_type[0], "_".join(backbone_type[1:])

    assert model_name in [
        "resnet18", "resnet50", "resnet101", "resnet152",
        "faster50",
    ], model_name

    assert weight_type in ["", "pt", "pt_frz"], weight_type
    # ""        =from scratch
    # "pt"      =pretrained+finetune
    # "pt_frz"  =pretrained+freeze

    use_torch_weight = (weight_type in ["pt", "pt_frz"]) and (weight_path is None)

    if "resnet" in model_name:
        # e.g.  torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        backbone = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=use_torch_weight)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Sequential()

    elif model_name == "faster50":
        backbone = FasterRCNN50Backbone(pretrained=use_torch_weight)
        feat_dim = backbone.representation_size

    else:
        raise NotImplementedError()

    # load weight with provided checkpoint
    if weight_type in ["pt", "pt_frz"] and weight_path:
        checkpoint = torch.load(weight_path)
        if "state_dict" in checkpoint:
            backbone.backbone.load_state_dict(checkpoint["state_dict"])
        else:
            backbone.backbone.load_state_dict(checkpoint)

    # freeze weight
    if weight_type == "pt_frz":
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, feat_dim


def onehot(x, depth, device):
    return torch.eye(depth)[x].to(device).float()


def build_counterfactual(causal, num_attr, num_aff):
    '''
    :param causal: [ N, 3 ] (inst_id, attr_id, aff_id)
    :param num_attr:
    :param num_aff:
    :return:
         counterfactual_inst_id : tensor [ M ]  index of instance in batch
         counterfactual_attr_mask: tensor [ M, num_attr ]  which attr to be skipped
         counterfactual_aff_mask: tensor [ M, num_aff ]  which aff will be affected after counterfactual
    '''
    orig_size = causal.shape[0]
    unique_inst_att_pair = torch.unique(causal[:, :2], dim=0)
    reduce_size = unique_inst_att_pair.shape[0]
    counterfactual_inst_id = unique_inst_att_pair[:, 0]
    counterfactual_attr_mask = onehot(unique_inst_att_pair[:, 1], num_attr, causal.device)
    space_mapping = torch.all(
        causal[:, :2].unsqueeze(0).expand(reduce_size, orig_size, 2) == \
        unique_inst_att_pair[:, :2].unsqueeze(1).expand(reduce_size, orig_size, 2),
        dim=2
    ).float()
    counterfactual_aff_mask = torch.matmul(space_mapping, onehot(causal[:, 2], num_aff, causal.device))

    return counterfactual_inst_id, counterfactual_attr_mask, counterfactual_aff_mask


class Aggregator(nn.Module):
    def __init__(self, method, args=None, num_para=None):
        super().__init__()
        self.support = ['sum', 'mean', 'max', 'concat']
        self.method = method

        if method not in self.support:
            raise NotImplementedError(
                'Not supported aggregation method [%s].\nWe only support: %s' % (method, self.support))

        if method == "concat":
            self.compression = nn.Linear(args.parallel_attr_rep_dim*num_para, args.aggr_rep_dim, bias=False)
            self.relu = nn.ReLU(inplace=True)

        if method == "qkv":
            raise NotImplementedError()

    def forward(self, tensor, mask=None, mask_method="zero"):
        """
        :param tensor:  bz * n * dim
        :param mask:    bz * n
        :return:        bz * dim
        """
        

        if mask is not None:
            if len(mask.size())==2:
                mask = mask.unsqueeze(-1)
            else:
                mask = mask.unsqueeze(-1).unsqueeze(0)

            if mask_method == "zero":
                tensor = tensor * mask
            elif mask_method == "random":
                rdm = torch.randn_like(tensor).to(tensor.device)
                tensor = torch.where(mask.expand_as(tensor), tensor, rdm)
            else:
                raise NotImplementedError(mask_method)

        if self.method == 'sum':
            return tensor.sum(1)
        elif self.method == 'mean':
            return tensor.mean(1)
        elif self.method == 'max':
            return tensor.max(1).values
        elif self.method == 'concat':
            out = tensor.reshape(tensor.shape[0], -1)
            out = self.compression(out)
            out = self.relu(out)
            return out



class FasterRCNN50Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(FasterRCNN50Backbone, self).__init__()

        self.backbone = backbone_utils.resnet_fpn_backbone('resnet50', pretrained)
        self.box_roi_pool = faster_rcnn.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

        resolution = self.box_roi_pool.output_size[0]
        out_channels = self.backbone.out_channels
        self.representation_size = 1024
        self.box_head = faster_rcnn.TwoMLPHead(
            out_channels * resolution ** 2,
            self.representation_size)

        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                faster_rcnn.model_urls['fasterrcnn_resnet50_fpn_coco'],
                progress=True)

            backbone_pref = "backbone."
            self.backbone.load_state_dict({
                k[len(backbone_pref):]: v
                for k, v in state_dict.items()
                if k.startswith(backbone_pref)
            })

            box_head_pref = "roi_heads.box_head."
            self.box_head.load_state_dict({
                k[len(box_head_pref):]: v
                for k, v in state_dict.items()
                if k.startswith(box_head_pref)
            })

    def forward(self, images, bboxes):
        image_sizes = [img.shape[-2:] for img in images]
        images = torch.stack(images, 0)

        features = self.backbone(images)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        box_features = self.box_roi_pool(features, bboxes, image_sizes)
        # input (Tensor[N, C, H, W]) – 输入张量
        # boxes (Tensor[K, 5] or List[Tensor[L, 4]]) – 输入的box 坐标，格式：list(x1, y1, x2, y2)或者(batch_index, x1, y1, x2, y2)
        # output_size (int or Tuple[int, int]) – 输出尺寸, 格式： (height, width)
        box_features = self.box_head(box_features)

        return box_features


class ParallelLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_para, bias=True):
        super(ParallelLinear, self).__init__()
        self.bias = bias
        self.__weight = nn.Parameter(torch.randn(
            num_para, in_dim, out_dim
        ))
        if self.bias:
            self.__bias = nn.Parameter(torch.zeros(
                num_para, out_dim
            ))

    def forward(self, x):
        x = torch.einsum('...ij, ijk -> ...ik', x, self.__weight)
        if self.bias:
            x = x + self.__bias
        return x



class ParallelMLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_para, hidden_layers=[], layernorm=True, bias=True, share_last_fc=False, out_relu=False):
        super().__init__()
        inner_bias = bias

        mod = []
        if hidden_layers is not None:
            last_dim = inp_dim
            for hid_dim in hidden_layers:
                mod.append(ParallelLinear(last_dim, hid_dim, num_para, bias=inner_bias))

                if layernorm:
                    mod.append(nn.LayerNorm(hid_dim))
                mod.append(nn.ReLU(inplace=True))
                last_dim = hid_dim

            if share_last_fc:
                mod.append(nn.Linear(last_dim, out_dim, bias=inner_bias))
            else:
                mod.append(ParallelLinear(last_dim, out_dim, num_para, bias=inner_bias))
            
            if out_relu:
                mod.append(nn.ReLU(inplace=True))

        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output



class PoolParallelMLP(ParallelMLP):
    def __init__(self, inp_dim, out_dim, num_in_chn, num_para, *args, **kwargs):
        super().__init__(inp_dim, out_dim, num_para, *args, **kwargs)
        self.pool = nn.Linear(num_in_chn, num_para)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        """
        (bz, nch, dim) -> (bz, npara, dim)
                       -> (bz, npara, dim_out)
        """
        x = x.transpose(-1, -2)
        x = self.pool(x)
        x = self.relu(x)
        x = x.transpose(-1, -2)
        output = self.mod(x)
        return output


class MLP(nn.Module):
    """Multi-layer perceptron, 1 layers as default. No activation after last fc"""

    def __init__(self, inp_dim, out_dim, hidden_layers=[], batchnorm=True, bias=True, out_relu=False, out_bn=False):
        super(MLP, self).__init__()

        inner_bias = bias and (not batchnorm)

        mod = []
        if hidden_layers is not None:
            last_dim = inp_dim
            for hid_dim in hidden_layers:
                mod.append(nn.Linear(last_dim, hid_dim, bias=inner_bias))
                if batchnorm:
                    mod.append(nn.BatchNorm1d(hid_dim))
                mod.append(nn.ReLU(inplace=True))
                last_dim = hid_dim

            mod.append(nn.Linear(last_dim, out_dim, bias=bias))
            if out_bn:
                mod.append(nn.BatchNorm1d(out_dim))
            if out_relu:
                mod.append(nn.ReLU(inplace=True))

        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output


class Distance(nn.Module):
    def __init__(self, metric):
        super(Distance, self).__init__()

        if metric == "L2":
            self.metric_func = lambda x, y: torch.norm(x - y, 2, dim=-1)
        elif metric == "L1":
            self.metric_func = lambda x, y: torch.norm(x - y, 1, dim=-1)
        elif metric == "cos":
            self.metric_func = lambda x, y: 1 - F.cosine_similarity(x, y, dim=-1)
        else:
            raise NotImplementedError("Unsupported distance metric: %s" % metric)

    def forward(self, x, y):
        output = self.metric_func(x, y)
        return output


class DistanceLoss(Distance):
    def forward(self, x, y):
        output = self.metric_func(x, y)
        output = torch.mean(output)
        return output


class TripletMarginLoss(Distance):
    def __init__(self, margin, metric):
        super(TripletMarginLoss, self).__init__(metric)
        self.triplet_margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = self.metric_func(anchor, positive)
        neg_dist = self.metric_func(anchor, negative)
        dist_diff = pos_dist - neg_dist + self.triplet_margin
        output = torch.max(dist_diff, torch.zeros_like(dist_diff).to(dist_diff.device))
        return output.mean()


class CrossEntropyLossWithProb(nn.Module):
    def __init__(self, weight=None, clip_thres=1e-8):
        super(CrossEntropyLossWithProb, self).__init__()
        self.nll = nn.NLLLoss(weight)
        self.clip_thres = clip_thres

    def forward(self, probs, labels):
        probs = probs.clamp_min(self.clip_thres)
        ll = torch.log(probs)
        return self.nll(ll, labels)


class CounterfactualHingeLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, cf_prob, orig_prob, gt_label, cf_label_mask):
        loss = torch.where(
            gt_label == 1,
            cf_prob - (orig_prob - self.margin),
            (orig_prob + self.margin) - cf_prob
        )
        # loss[loss < 0] = 0
        loss = nn.functional.relu(loss, inplace=True)

        loss = loss * cf_label_mask
        loss = loss.mean(0).sum()
        return loss




class OcrnBaseModel(nn.Module):

    def __init__(self, dataset, args):
        super(OcrnBaseModel, self).__init__()

        self.args = args
        self.num_obj = len(dataset.objs)
        self.num_attr = len(dataset.attrs)
        self.num_aff = dataset.num_aff

        # submodules
        if args.data_type == "feature":
            self.backbone = None
            self.feat_dim = dataset.feature_dim
        else:
            self.backbone, self.feat_dim = load_backbone(args.backbone_type, args.backbone_weight)

        # prior information
        prior_info = torch.load(f"features/OCL_{args.backbone_type}/obj_prior.t7")
        self.register_buffer("mean_obj_features",
            prior_info["mean_obj_features"] )  # (n_obj, dim)

        
        # preproc P(O)
        if args.obj_prior_type == "default":
            pass
        elif args.obj_prior_type == "step":
            sep = np.linspace(0, self.num_obj, args.obj_prior_bins, dtype=int).tolist()
            frequency = prior_info["freqency"].numpy()
            order = frequency.argsort()
            for i,j in zip(sep[:-1], sep[1:]):
                ids = order[i:j]
                frequency[ids] = frequency[ids].mean()
            prior_info["freqency"] = torch.from_numpy(frequency)
        else:
            raise NotImplementedError(args.obj_prior_type)

        self.register_buffer("obj_frequence", 
            prior_info["freqency"] )  # (n_obj,)
        assert len(prior_info["freqency"].size())==1
        

        CA = json.load(open('data/resources/OCL_category_annot.json'))
        self.register_buffer("category_attr",
            torch.Tensor([ CA[o]['attr'] for o in dataset.objs ]).float() )
        self.register_buffer("category_aff",
            torch.Tensor([ CA[o]['aff'] for o in dataset.objs ]).float() )

        print(f"CA: attr={self.category_attr.shape}, aff={self.category_aff.shape}")

        # loss weight
        if args.loss_class_weight:
            class_weight = json.load(open("data/resources/OCL_weight.json"))
            self.register_buffer("obj_loss_wgt",  torch.tensor(class_weight["obj_weight"]))
            self.register_buffer("attr_loss_wgt", torch.tensor(class_weight["attr_weight"]))
            self.register_buffer("aff_loss_wgt",  torch.tensor(class_weight["aff_weight"]))
        else:
            self.obj_loss_wgt, self.attr_loss_wgt, self.aff_loss_wgt = None, None, None

        self.pos_weight_attr = None
        self.pos_weight_aff = None
    
    

        # losses
        if args.positive_bce:
            self.attr_bce = PositiveBCELoss(class_weight=self.attr_loss_wgt)
            self.aff_bce = PositiveBCELoss(class_weight=self.aff_loss_wgt)
        else:
            self.attr_bce = nn.BCEWithLogitsLoss(weight=self.attr_loss_wgt, pos_weight=self.pos_weight_attr)
            self.aff_bce = nn.BCEWithLogitsLoss(weight=self.aff_loss_wgt, pos_weight=self.pos_weight_aff)
        
        self.pair_prob_bce = nn.BCELoss()
