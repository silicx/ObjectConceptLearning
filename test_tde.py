_ = f"If you see this message in SyntaxError, you are using an older Python environment (>=3.8 required)"
import torch

torch.backends.cudnn.benchmark = True

import os
import numpy as np
import tqdm
import logging
logging.basicConfig(format='[%(asctime)s] %(name)s: %(message)s', level=logging.INFO)
import importlib
from collections import defaultdict
from utils import dataset, utils
from utils.evaluator import tde_evaluator, mAP_evaluator


def main():
    logger = logging.getLogger('MAIN')

    # read cmd args
    args = utils.parse_ocrn_args()
    utils.display_args(args, logger)
    network_module = importlib.import_module('models.' + args.network)

    logger.info("Loading dataset")

    feature_dir = f"features/OCL_{args.backbone_type}"
    test_dataloader = dataset.get_dataloader(
        'valtest', args.data_type, feature_dir,
        batchsize=args.bz, num_workers=args.num_workers)

    logger.info("Loading network")
    model = network_module.TDEModel(test_dataloader.dataset, args)
    model = model.to(args.device)

    # initialization (model weight, optimizer, lr_scheduler, clear logs)
    model, checkpoint = utils.initialize_model(model, args)

    # val
    logger.info('Start eval tde')

    epoch = 0 if checkpoint is None else checkpoint['epoch']

    current_reports = test_epoch_tde(model, test_dataloader, args, epoch)

    # print test results
    for key, value in current_reports.items():
        value["epoch"] = epoch
        utils.color_print(f"{args.name} {key}" + utils.formated_ocl_result(value))


##########################################################################

@torch.no_grad()
def test_epoch_tde(model, dataloader, args, epoch):
    all_gt = defaultdict(list)
    all_pred = defaultdict(list)
    val_mask = []

    for _, batch in tqdm.tqdm(enumerate(dataloader),
                              total=len(dataloader),
                              postfix='Test',
                              ncols=75,
                              leave=False):

        feed_batch = utils.batch_to_device({
            "image": batch["image"],
            'gt_attr': batch['gt_attr']
        }, args.device)
        preds = model(feed_batch, require_loss=False)

        for k, v in preds.items():
            all_pred[k].append(v.detach().cpu())

        for key in ['gt_attr', 'gt_aff', 'gt_causal', 'val_mask']:
            if isinstance(batch[key], list):
                batch[key] = torch.cat(batch[key], 0)

        all_gt['attr'].append(batch['gt_attr'])
        all_gt['aff'].append(batch['gt_aff'])

        bz = batch['gt_attr'].shape[0]
        causal_matrix = torch.zeros((bz, batch['gt_attr'].shape[1], batch['gt_aff'].shape[1]))
        for inst_id, attr_id, aff_id in batch['gt_causal']:
            causal_matrix[inst_id, attr_id, aff_id] = 1

        all_gt['causal'].append(causal_matrix)
        val_mask.append(batch['val_mask'])

    all_gt = {k: torch.cat(v, 0).numpy() for k, v in all_gt.items()}
    all_pred = {k: torch.cat(v, 0).numpy() for k, v in all_pred.items()}
    val_mask = torch.cat(val_mask, 0).numpy()

    # concatenate gt/det items (and filter with val_mask)
    if args.save_file is not None:
        import h5py
        f = h5py.File(args.save_file, 'w')
        for k, v in all_gt.items():
            f[k] = v
        for k, v in all_pred.items():
            f[k] = v
        f['val_mask'] = val_mask
    val_res = evaluate_tde(all_gt, all_pred, val_mask)
    test_res = evaluate_tde(all_gt, all_pred, ~val_mask)
    results = [val_res, test_res]
    name_prefix = ['val_', 'test_']

    all_reports = {}
    for name in val_res.keys():
        # additional eval scores
        report_dict = {
            'epoch': epoch,
        }

        for res, pref in zip(results, name_prefix):
            for key, value in res[name].items():
                report_dict[pref + key] = value

        all_reports[name] = report_dict

    return all_reports


def evaluate_tde(all_gt, all_pred, masks=None):
    assert 'attr_main' in all_pred
    assert 'aff_main' in all_pred
    assert 'tde_main' in all_pred
    if masks is not None:
        all_gt = {k: v[masks, ...] for k, v in all_gt.items()}
        all_pred = {k: v[masks, ...] for k, v in all_pred.items()}

    report = {}
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
            report[name] = report_dict

        elif name.startswith("aff"):
            report_dict = {
                'mAP': mAP_evaluator(pred, all_gt["aff"]),
            }
            report[name] = report_dict

        elif name.startswith("attr"):
            report_dict = {
                'mAP': mAP_evaluator(pred, all_gt["attr"]),
            }
            report[name] = report_dict
        elif name.startswith('tde'):
            report['tde_main'] = tde_evaluator(all_pred, all_gt)
        else:
            raise NotImplementedError()

    return report


if __name__ == "__main__":
    main()
