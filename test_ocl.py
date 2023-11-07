_ = f"If you see this message in SyntaxError, you are using an older Python environment (>=3.8 required)"
import torch

torch.backends.cudnn.benchmark = True

import numpy as np
import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.basicConfig(format='[%(asctime)s] %(name)s: %(message)s', level=logging.INFO)
import importlib
from collections import defaultdict

from utils import dataset, utils
from utils.evaluator import mAP_evaluator
import random


def main():
    logger = logging.getLogger('MAIN')

    # read cmd args
    args = utils.parse_ocrn_args()
    utils.display_args(args, logger)
    network_module = importlib.import_module('models.' + args.network)

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logger.info("Loading dataset")

    feature_dir = f"features/OCL_{args.backbone_type}"
    test_dataloader = dataset.get_dataloader(
        'valtest', args.data_type, feature_dir,
        batchsize=args.bz, num_workers=args.num_workers)

    logger.info("Loading network and optimizer")
    model = network_module.Model(test_dataloader.dataset, args)
    assert torch.cuda.device_count() == 1

    model = model.to(args.device)
    print(model)

    # initialization (model weight, optimizer, lr_scheduler, clear logs)

    model, checkpoint = utils.initialize_model(model, args)
    init_epoch = checkpoint["epoch"] if checkpoint is not None else 0


    # trainval
    logger.info('Start evaluating')
    logger.info('Origin result')
    
    current_reports, current_scores = test_epoch(model, test_dataloader, init_epoch, args)  # [num_att]





@torch.no_grad()
def test_epoch(model, dataloader, epoch, args):
    all_gt = defaultdict(list)
    all_pred = defaultdict(list)
    val_mask = []

    for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), postfix='Test %d' % epoch, ncols=75,
                              leave=False):

        feed_batch = utils.batch_to_device({
            "image": batch["image"],
            "gt_attr": batch["gt_attr"],
            "gt_aff": batch["gt_aff"],
            "gt_obj_id": batch["gt_obj_id"],
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

        all_reports[name] = report_dict

    for key, value in all_reports.items():
        utils.color_print(f"{args.name} {key}" + utils.formated_ocl_result(value))
    
    after_pred = {k: v.mean(0) for k, v in all_pred.items()}
    return after_pred, all_reports


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
    main()
