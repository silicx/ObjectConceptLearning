import argparse
import functools
from typing import Dict, Tuple
import re
import os
import shutil
import logging
import os.path as osp
from easydict import EasyDict
from termcolor import colored
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import yaml





def display_args(args, logger, verbose=False):
    """print some essential arguments"""
    if verbose:
        ignore = []
        for k, v in args.__dict__.items():
            if not callable(v) and not k.startswith('__') and k not in ignore:
                logger.info("{:30s}{}".format(k, v))
    else:
        logger.info('Name:       %s' % args.name)
        logger.info('Network:    %s' % args.network)


def duplication_check(args, log_dir):
    if args.force:
        return
    elif args.trained_weight is None:
        assert not osp.exists(log_dir), "log dir with same name exists (%s)" % log_dir


def formated_ocl_result(report: dict):
    def prior_to(s1, s2):
        if "hard" in s1 and "hard" not in s2:
            return True
        elif "mAP" in s1 and "mAP" not in s2:
            return True
        elif "t_" in s1 and "t_" not in s2:
            return True

    def comparator(s1, s2):
        if prior_to(s1, s2):
            return -1
        elif prior_to(s2, s1):
            return 1
        else:
            return 0

    fstr = '[E {epoch}]'
    keylist = [x for x in report.keys() if x not in ["epoch", "name"]]
    keylist = sorted(keylist, key=functools.cmp_to_key(comparator))
    for key in keylist:
        fstr += ' %s:{%s:.4f}' % (key, key)
    return fstr.format(**report)



################################################################################
#                                network utils                                 #
################################################################################

def repeat_on(tensor: torch.Tensor, repeats: int, dim: int) -> torch.Tensor:
    return tensor.repeat_interleave(repeats, dim)


def tile_on(tensor: torch.Tensor, repeats: int, dim: int) -> torch.Tensor:
    repvec = [1] * len(tensor.size())
    repvec[dim] = repeats
    return tensor.repeat(*repvec)
    # torch>=1.8
    # return torch.tile(tensor, tuple(repvec))


def activation_func(name: str):
    if name == "none":
        return nn.Sequential()
    elif name == "sigmoid":
        return torch.sigmoid
    elif name == "relu":
        return torch.ReLU(inplace=True)
    else:
        raise NotImplementedError("Activation function {} is not implemented".format(name))


def get_optimizer(optim_type, lr, params):
    if optim_type != 'sgd':
        logger = logging.getLogger('utils.get_optimizer')
        logger.info('Using {} optimizer'.format(optim_type))
    lr = float(lr)
    if optim_type == 'sgd':
        return torch.optim.SGD(params, lr=lr)
    elif optim_type == 'momentum':
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif optim_type == 'adam':
        return torch.optim.Adam(params, lr=lr)
    elif optim_type == 'adamw':
        return torch.AdamW(params, eps=5e-5, lr=lr)
    elif optim_type == 'rmsprop':
        return torch.RMSprop(params, lr=lr)
    else:
        raise NotImplementedError("{} optimizer is not implemented".format(optim_type))


def set_scheduler(args, optimizer, train_dataloader):
    """ return a learning rate scheduler """

    if args.lr_decay_type == 'no' or args.lr_decay_type is None:
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=(lambda epoch: 1.0))
    elif args.lr_decay_type == 'exp':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=len(train_dataloader) * args.lr_decay_step, gamma=args.lr_decay_rate)
    elif args.lr_decay_type == 'cos':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=len(train_dataloader) * args.lr_decay_step, T_mult=2, eta_min=0)
    elif args.lr_decay_type == 'multi step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, [val * len(train_dataloader) for val in args.lr_decay_step], gamma=args.lr_decay_rate)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented' % args.lr_decay_type)
    return scheduler


def clear_folder(dirname: str):
    """clear weight and log dir"""
    logger = logging.getLogger('utils.clear_folder')

    for f in os.listdir(dirname):
        logger.warning('Deleted log file ' + f)
        os.remove(os.path.join(dirname, f))


class CheckpointPath(object):
    TEMPLATE = "{:s}/checkpoint_ep{:d}.pt"
    EPOCH_PATTERN = "checkpoint_ep([0-9]*).pt"

    def compose(log_dir: str, epoch: int) -> str:
        return CheckpointPath.TEMPLATE.format(log_dir, epoch)

    def decompose(ckpt_path: str, ) -> Tuple[str, int]:
        log_dir = osp.dirname(ckpt_path)
        re_groups = re.match(CheckpointPath.EPOCH_PATTERN, osp.basename(ckpt_path))
        if re_groups:
            epoch = int(re_groups.group(1))
        else:
            epoch = None
        return log_dir, epoch

    def in_dir(ckpt_path: str, log_dir: str) -> bool:
        ckpt_log_dir, _ = CheckpointPath.decompose(ckpt_path)
        return osp.samefile(ckpt_log_dir, log_dir)


def generate_pair_result(pred_attr, pred_obj, dset):
    scores = {}
    for i, (attr, obj) in enumerate(dset.pairs):
        attr = dset.attr2idx[attr]
        obj = dset.obj2idx[obj]
        scores[(attr, obj)] = pred_attr[:, attr] * pred_obj[:, obj]
    return scores


def dict_to_namespace(dict_var: Dict) -> EasyDict:
    for key, value in dict_var.items():
        if isinstance(value, dict):
            # recursively convert to namespace
            dict_var[key] = dict_to_namespace(value)
    return EasyDict(dict_var)


def namespace_to_dict(nspace_var: EasyDict) -> Dict:
    for key in nspace_var:
        if isinstance(nspace_var[key], dict):
            nspace_var[key] = namespace_to_dict(nspace_var[key])
    return dict(nspace_var)


def parse_ocrn_args():
    def override_dict(source_dict: Dict, override_dict: Dict):
        """merge two dictionary recursively (the source dictionary may be altered).
        If same keys occur, the value of the latter will override that of the former.
        Different from `dict.update()` due to the """
        for key, value in override_dict.items():
            if key in source_dict and isinstance(source_dict[key], dict):
                # recursively override the content
                if not isinstance(value, dict):
                    raise ValueError(f"Cannot override key {key}: the value should have `dict` type.")
                source_dict[key] = override_dict(source_dict[key], value)
            else:
                source_dict[key] = value
        return source_dict

    parser = argparse.ArgumentParser()
    # basic configs
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--force", default=False, action="store_true",
                        help="WARINING: clear experiment with duplicated name")
    parser.add_argument("--amp", default=False, action="store_true",
                        help="amp training")
    parser.add_argument("--trained_weight", type=str, default=None,
                        help="Restore from a certain trained weight (relative path to './weights')")
    parser.add_argument("--weight_type", type=str, default=None,
                        help="Type of the trained weight: 'continue'-previous checkpoint(default)")
    parser.add_argument("--box_file", type=str,
                        help="None/faster/fsdet, or a COCO style detection result file")
    # training / debug / test args
    parser.add_argument("--save_file", default=None, type=str)
    parser.add_argument("--splits", type=str, help="For feature extraction")
    parser.add_argument("--iou_thres", type=float, help="For det test. default=[0.3, 0.5]")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--gt_test", default=False, action="store_true", help="For det test")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--freeze_type', type=str)
    # override configs
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--bz", type=int)
    cmd_args = parser.parse_args()

    with open("config/BaseConfig.yaml") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    with open(cmd_args.cfg) as fp:
        config = override_dict(config, yaml.load(fp, Loader=yaml.FullLoader))

    if cmd_args.name is None:
        cmd_args.name = osp.splitext(osp.basename(cmd_args.cfg))[0]

    cmd_args = vars(cmd_args)
    args_to_override = ["epoch", "lr", "bz", "trained_weight", 'weight_type', 'freeze_type']
    config.update({k: cmd_args[k] for k in args_to_override if cmd_args[k] is not None})
    config.update({k: v for k, v in cmd_args.items() if k not in args_to_override})


    os.makedirs(config['log_dir'], exist_ok=True)

    # default args
    if "parallel_attr_rep_dim" not in config:
        config["parallel_attr_rep_dim"] = config["attr_rep_dim"]
    if "parallel_aff_rep_dim" not in config:
        config["parallel_aff_rep_dim"] = config["aff_rep_dim"]
    if "aggr_rep_dim" not in config:
        config["aggr_rep_dim"] = config["parallel_attr_rep_dim"]
    if "attr_hidden_rep_dim" not in config:
        config["attr_hidden_rep_dim"] = config["attr_rep_dim"]
    if "aff_hidden_rep_dim" not in config:
        config["aff_hidden_rep_dim"] = config["aff_rep_dim"]
    if "attr_out_rep_dim" not in config:
        config["attr_out_rep_dim"] = config["attr_rep_dim"]
    if "aff_out_rep_dim" not in config:
        config["aff_out_rep_dim"] = config["aff_rep_dim"]


    args = dict_to_namespace(config)
    return args


def snapshot(model, optimizer, lr_scheduler, epoch, log_dir):
    """save checkpoint"""
    ckpt_path = CheckpointPath.compose(log_dir, epoch)

    current_disk = "/" + os.path.realpath(log_dir).split("/")[1]
    _, _, free = shutil.disk_usage(current_disk)
    free = free / (2**30) # GB
    if free < 5.0:
        logging.warning(f"Skip saving checkpoint: {current_disk} is almost full ({free} GB remaining)")
        return 

    if isinstance(model, nn.DataParallel):
        torch.save({
            "epoch": epoch,
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }, ckpt_path)
    else:
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }, ckpt_path)

    logging.getLogger('utils.snapshot').info('Wrote snapshot to: {}'.format(ckpt_path))


def initialize_model(model: nn.Module, args) -> nn.Module:
    logger = logging.getLogger('initialize')

    if args.trained_weight:

        # Compatible with tf version 
        args.weight_type = {
            None : 'continue',
            "1": "continue",
            "2": 'pretrain',
            "3": 'pretrain_bug'
        }.get(args.weight_type, args.weight_type)

        logger.info(f"Restoring backbone from {args.trained_weight} ({args.weight_type} )")

        if args.distributed:
            checkpoint = torch.load(args.trained_weight, map_location={f'cuda:0': f'cuda:{args.local_rank}'} )
        else:
            checkpoint = torch.load(args.trained_weight)

        state_dict = checkpoint["state_dict"]
        if all([x.startswith("module.") for x in state_dict.keys()]):
            state_dict = {k[7:]:v for k,v in state_dict.items()}
        

        if args.weight_type == "continue":
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            logger.info('Missing: %s' % missing_keys)
            logger.info('Unexpected: %s' % unexpected_keys)
            return model, checkpoint
        elif args.weight_type == 'pretrain':
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            logger.info('Missing: %s' % missing_keys)
            logger.info('Unexpected: %s' % unexpected_keys)
            return model, None
        elif args.weight_type in ['ocrn_attr', 'ocrn_aff', 'ocrn_para_attr', 'ocrn_para_aff']:
            param_names = get_params_group(model, args.weight_type)
            missing_keys, unexpected_keys = model.load_state_dict({
                k:v for k,v in state_dict.items() if k in param_names
            }, strict=False)
            logger.info('Missing: %s' % missing_keys)
            logger.info('Unexpected: %s' % unexpected_keys)
            return model, None
            
        else:
            raise NotImplementedError(args.weight_type)

    return model, None


def batch_to_device(batch, device):
    """recursively move Tensor elements to cuda"""
    if torch.is_tensor(batch):
        batch = batch.to(device)
    elif isinstance(batch, dict):
        for k in batch:
            batch[k] = batch_to_device(batch[k], device)
    elif isinstance(batch, list):
        for k in range(len(batch)):
            batch[k] = batch_to_device(batch[k], device)
    return batch


def get_params_group(model, group_name):

    group = set()

    if group_name == "ocrn_attr":
        keys = [
            "fc_feat2attr", "attr_instantialize", 
            "parallel_attr_feat", "attr_IA_classifier", "attr_CA_classifier",
        ]
    elif group_name == "ocrn_aff":
        keys = [
            "fc_feat2aff", "aff_instantialize", "aggregator", 
            "parallel_aff_feat", "aff_IA_classifier", "aff_CA_classifier",
        ]
    elif group_name == "ocrn_para_attr":
        keys = [
            "fc_feat2attr", "attr_instantialize", 
            "attr_IA_classifier", "attr_CA_classifier",
        ]
    elif group_name == "ocrn_para_aff":
        keys = [
            "parallel_attr_feat", "attr_auxIA_classifier", "aggregator", 
            "fc_feat2aff", "aff_instantialize",
            "aff_IA_classifier", "aff_CA_classifier",
        ]

    else:
        raise NotImplementedError(group_name)
    
    for name in model.state_dict():
        for key in keys:
            if name.startswith(key):
                group.add(name)
                break

    return group


def freeze_params(model, freeze_type):
    if freeze_type is not None:
        params_to_freeze = get_params_group(model, freeze_type)
        for name in params_to_freeze:
            model.state_dict()[name].requires_grad = False
            params_to_freeze.add(name)
            print("freeze", name)
    
        return params_to_freeze
    else:
        return set()



def color_print(string, color='green'):
    print(colored(string, color))
