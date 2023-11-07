_ = f"If you see this message in SyntaxError, you are using an older Python environment (>=3.8 required)"
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
import os
import os.path as osp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.basicConfig(format='[%(asctime)s] %(name)s: %(message)s', level=logging.INFO)

import numpy as np
import tqdm
import importlib
import yaml
import copy
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

from utils import dataset, utils
from utils.evaluator import mAP_evaluator
import random



def main():
    # read cmd args
    args = utils.parse_ocrn_args()
    logger = logging.getLogger(f'MAIN#{args.local_rank}')
    network_module = importlib.import_module('models.' + args.network)

    # setup
    world_size = int(os.environ.get('WORLD_SIZE', 0))
    args.distributed =  (world_size > 1)
    assert args.distributed


    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", rank=args.local_rank)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # logging and pretrained weight dirs
    if args.local_rank == 0:
        logger.info(f"World size = {world_size}")
        
        utils.display_args(args, logger)
        log_dir = osp.join(args.log_dir, args.name)
        utils.duplication_check(args, log_dir)
        logger.info("Training ckpt and log  => " + log_dir)
        os.makedirs(log_dir, exist_ok=True)


    # loading data
    logger.info("Loading dataset")
    feature_dir = f"features/OCL_{args.backbone_type}"
    train_dataloader = dataset.get_dataloader(
        'train', args.data_type, feature_dir,
        batchsize=args.bz, num_workers=args.num_workers, distributed_sampler=True)
    test_dataloader = dataset.get_dataloader(
        'valtest', args.data_type, feature_dir,
        batchsize=args.bz, num_workers=args.num_workers, distributed_sampler=False)

    logger.info("Loading network and optimizer")
    model = network_module.Model(train_dataloader.dataset, args)
    freeze_param_names = utils.freeze_params(model, args.freeze_type)
    if args.lr_groups is None:
        optimizer = utils.get_optimizer(
            args.optimizer, args.lr, 
            [p for name, p in model.named_parameters() if name not in freeze_param_names])
    else:
        param_groups = []
        used_param_names = copy.copy(freeze_param_names) # already assigned lr
        for info in args.lr_groups:
            param_name = {x for x in utils.get_params_group(model, info["name"])
                            if x not in freeze_param_names}
            used_param_names.update(param_name)
            param_groups.append({
                "params": [p for name, p in model.named_parameters() if name in param_name],
                "lr": info["lr"],
            })
            if args.local_rank==0:
                print(f"lr {info['lr']}: ", {name.split(".")[0] for name, p in model.named_parameters() if name in param_name})

        default_lr_params = [p for name, p in model.named_parameters() if name not in used_param_names]
        
            
        if len(default_lr_params)>0:
            param_groups.append({"params": default_lr_params})
            if args.local_rank==0:
                print("dafault lr: ", {name.split(".")[0] for name, _ in model.named_parameters() if name not in used_param_names})

        optimizer = utils.get_optimizer(
            args.optimizer, args.lr, param_groups)

    if args.local_rank == 0:
        print(model)
    scheduler = utils.set_scheduler(args, optimizer, train_dataloader)

    # initialization (model weight, optimizer, lr_scheduler, clear logs)
    if args.local_rank==0 and (args.trained_weight is None or args.weight_type != "continue"):
        utils.clear_folder(log_dir)  # --force

    model = model.to(args.local_rank)
    model, checkpoint = utils.initialize_model(model, args)
    init_epoch = 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        init_epoch = checkpoint["epoch"]

    if args.distributed:   # after load_state_dict
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # evaluator
    main_score_name = "aff_main"
    main_score_key = 'val_mAP'
    best_reports = None

    # logger
    if args.local_rank == 0:
        writer = SummaryWriter(log_dir)
        config_path = osp.join(log_dir, "config.yaml")
        with open(config_path, "w") as fp:
            yaml.dump(utils.namespace_to_dict(args), fp)
            logger.info(f"Configs are saved to {config_path}")
    else:
        writer = None

    # trainval
    logger.info('Start training')


    for epoch in range(init_epoch + 1, args.epoch + 1):

        train_epoch(model, optimizer, scheduler, train_dataloader, writer, epoch, args)

        if args.local_rank == 0:
            if args.test_freq > 0 and epoch % args.test_freq == 0:
                current_reports = test_epoch(model, test_dataloader, writer, epoch, args)

                if (best_reports is None or
                        current_reports[main_score_name][main_score_key] > best_reports[main_score_name][main_score_key]
                ):
                    best_reports = current_reports

                # print test results
                for key, value in current_reports.items():
                    utils.color_print(f"{args.name} Current {key}" + utils.formated_ocl_result(value))
                for key, value in best_reports.items():
                    utils.color_print(f"{args.name} Best {key}" + utils.formated_ocl_result(value))

            if args.snapshot_freq > 0 and epoch % args.snapshot_freq == 0:
                utils.snapshot(model, optimizer, scheduler, epoch, log_dir)


    if args.local_rank == 0:
        writer.close()
    logger.info('Finished.')


##########################################################################

def train_epoch(model, optimizer, scheduler, dataloader, writer, epoch, args):
    model.train()

    if args.local_rank==0:
        progressor = lambda x: tqdm.tqdm(x, total=len(dataloader),
                                      postfix='Train %d/%d' % (epoch, args.epoch), ncols=75, leave=False)
    else:
        progressor = lambda x:x

    summary_interval = 50 # int(len(dataloader) * 0.1)
    sum_loss = defaultdict(list)

    def _write_to_tb(ind):
        """write losses to tensorboard (mean value of recent iters)"""
        for key, value in sum_loss.items():
            if writer:
                writer.add_scalar(
                    key,
                    np.mean(value[-summary_interval:]),
                    (epoch-1) * len(dataloader) + ind
                )

    for batch_ind, batch in progressor(enumerate(dataloader)):

        feed_batch = utils.batch_to_device({
            "image": batch["image"],
            "gt_attr": batch["gt_attr"],
            "gt_aff": batch["gt_aff"],
            "gt_obj_id": batch["gt_obj_id"],
            "gt_causal": batch["gt_causal"],
            "main_bbox": batch.get("main_bbox", None),
        }, args.local_rank)
        losses = model(feed_batch)

        if 'loss_total' in losses:
            optimizer.zero_grad()
            loss_total = losses['loss_total']
            loss_total.backward()
                
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

        for key, value in losses.items():
            sum_loss[key].append(value.sum().item())

        if args.local_rank == 0:
            if (batch_ind + 1) % summary_interval == 0:
                _write_to_tb(batch_ind)
                
    _write_to_tb(len(dataloader))

    train_str = f'Train Epoch {epoch}: lr [ '
    for group in optimizer.param_groups:
        train_str += "%.2e "%group['lr']
    train_str += "] "
    for key in sum_loss.keys():
        v = np.mean(sum_loss[key])
        train_str += '%s %.2f ' % (key, v)
    print(train_str)


@torch.no_grad()
def test_epoch(model, dataloader, writer, epoch, args):
    all_gt = defaultdict(list)
    all_pred = defaultdict(list)
    val_mask = []

    for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), postfix='Test %d' % epoch, ncols=75,
                              leave=False):

        feed_batch = utils.batch_to_device({
            "image": batch["image"],
            "main_bbox": batch.get("main_bbox", None),
        }, args.device)
        preds = model(feed_batch, require_loss=False)
        for k, v in preds.items():
            all_pred[k].append(v.detach().cpu())

        for key in ['gt_attr', 'gt_aff', 'gt_obj_id', 'val_mask']:
            if isinstance(batch[key], list):
                batch[key] = torch.cat(batch[key], 0)

        all_gt['attr'].append(batch['gt_attr'])
        all_gt['aff'].append(batch['gt_aff'])
        all_gt['obj'].append(batch['gt_obj_id'])
        val_mask.append(batch['val_mask'])

    all_gt = {k: torch.cat(v, 0).numpy() for k, v in all_gt.items()}
    all_pred = {k: torch.cat(v, 0).numpy() for k, v in all_pred.items()}
    val_mask = torch.cat(val_mask, 0).numpy()

    val_res = evaluate_joint(all_gt, all_pred, val_mask)
    test_res = evaluate_joint(all_gt, all_pred, ~val_mask)
    results = [val_res, test_res]
    name_prefix = ['val_', 'test_']

    all_reports = {}
    for name in all_pred:
        # additional eval scores
        report_dict = {
            'epoch': epoch,
        }

        for res, pref in zip(results, name_prefix):
            for key, value in res[name].items():
                report_dict[pref + key] = value

        # save to tensorboard
        if writer:
            for key, value in report_dict.items():
                if key not in ['name', 'epoch']:
                    writer.add_scalar(f"{name}/{key}", value, epoch)

        all_reports[name] = report_dict

    return all_reports


def evaluate_joint(all_gt, all_pred, instance_mask, return_vec=False):
    report = {}
    if instance_mask is not None:
        all_gt = {k: v[instance_mask, ...] for k, v in all_gt.items()}
        all_pred = {k: v[instance_mask, ...] for k, v in all_pred.items()}

    for name, pred in all_pred.items():
        # print(name)
        report_dict = None

        if name.startswith('obj'):
            # evaluate obj
            pred = np.argmax(pred, axis=1)  # pred obj id
            assert pred.shape == all_gt["obj"].shape
            acc = np.mean(pred == all_gt["obj"])

            report_dict = {
                'ACC': acc
            }

        elif name.startswith("aff"):
            report_dict = {
                'mAP': mAP_evaluator(pred, all_gt["aff"]),
            }

        elif name.startswith("attr"):
            report_dict = {
                'mAP': mAP_evaluator(pred, all_gt["attr"], return_vec=return_vec),
            }
        else:
            raise NotImplementedError()

        report[name] = report_dict

    return report


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt as e:
        torch.distributed.destroy_process_group()
        raise e
