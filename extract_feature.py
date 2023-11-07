_ = f"If you see this message in SyntaxError, you are using a older Python environment (>=3.8 required)"
import torch
torch.backends.cudnn.benchmark = True

import gc
import tqdm
import os, glob
import os.path as osp
import logging
import importlib

from utils import dataset, utils



def main():
    logger = logging.getLogger('MAIN')

    # read cmd args
    args = utils.parse_ocrn_args()
    utils.display_args(args, logger)
    


    logger.info("Loading dataset")

    if args.splits is None or args.splits=="all":
        splits_to_extract = ["train", "valtest"]
        feature_dir = f"features/OCL_{args.backbone_type}/"
    else:
        splits_to_extract = [args.splits]
        
        if args.splits == "det_valtest":
            box_name = osp.splitext(osp.basename(args.box_file))[0]
            feature_dir = f"features/OCL_{args.backbone_type}_{box_name}/"
        else:
            feature_dir = f"features/OCL_{args.backbone_type}/"

    dataloaders = {}
    for split_name in splits_to_extract:
        dataloaders[split_name] = dataset.get_dataloader(
            split_name, batchsize=args.bz,
            data_type=args.data_type, shuffle=False, num_workers=args.num_workers)

    logger.info("Loading network and optimizer")
    network_module = importlib.import_module('models.' + args.network)
    model = network_module.Model(list(dataloaders.values())[0].dataset, args)
    model = model.to(args.device)
    model, _ = utils.initialize_model(model, args)


    # eval
    logger.info('Start eval')
    os.makedirs(feature_dir, exist_ok=True)


    with torch.no_grad():
        for split_name, loader in dataloaders.items():
            feat_file = osp.join(feature_dir, f"{split_name}.t7")
            if not args.force and osp.exists(feat_file):
                logger.info(f"File {feat_file} exists, skip.")
            else:
                parts_dir = osp.join(feature_dir, "tmp_parts")
                os.makedirs(parts_dir, exist_ok=True)
                for data, begin, end in test_epoch_generator(model, loader, args.device, 500):
                    feat_file_part = osp.join(parts_dir, f"{split_name}.t7.part{begin}_{end}")
                    assert not osp.exists(feat_file_part)
                    torch.save(data, feat_file_part)
                    logger.info(f"Saved to {feat_file_part}")

                
                features = []
                file_names = []
                for fpath in glob.glob(osp.join(parts_dir, f"{split_name}.t7.part*")):
                    data = torch.load(fpath)
                    features += data["features"]
                    file_names += data["file_names"]
                torch.save({
                    "features": features,
                    "file_names": file_names,
                }, feat_file)



    logger.info('Finished.')



def test_epoch(model, dataloader, device):
    features = []
    file_names = []

    for index, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), postfix=f'Eval', ncols=75):
        if index%100 == 0:
            gc.collect()

        assert len(batch["file_name"])==1 and len(batch["image"])==1, str(batch["file_name"])

        feed_batch = utils.batch_to_device({
            "image": batch["image"],
            "main_bbox": batch["main_bbox"],
        }, device)

        if feed_batch["main_bbox"][0].size(0) > 0:
            # print(feed_batch["main_bbox"][0].size(0), feed_batch["image"][0].size())
            f = model(feed_batch)
            f = f.detach().cpu()
            assert len(f.size())==2, str(f.size())
            features.append(f)
            file_names.append(batch["file_name"][0])
        
        del feed_batch

    return {
        "features": features,
        "file_names": file_names,
    }




def test_epoch_generator(model, dataloader, device, part_size):
    features = []
    file_names = []
    index_begin = 0

    for index, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), postfix=f'Eval', ncols=75):

        if index%10 == 0:
            gc.collect()

        assert len(batch["file_name"])==1 and len(batch["image"])==1, str(batch["file_name"])

        feed_batch = utils.batch_to_device({
            "image": batch["image"],
            "main_bbox": batch["main_bbox"],
        }, device)

        if feed_batch["main_bbox"][0].size(0) > 0:
            # print(feed_batch["main_bbox"][0].size(0), feed_batch["image"][0].size())
            f = model(feed_batch)
            f = f.detach().cpu()
            assert len(f.size())==2, str(f.size())
            features.append(f)
            file_names.append(batch["file_name"][0])
        
        del feed_batch

        if (index+1) % part_size == 0:

            yield ({
                "features": features,
                "file_names": file_names,
            }, index_begin+1, index+1)

            index_begin = index
            features = []
            file_names = []

    yield ({
        "features": features,
        "file_names": file_names,
    }, index_begin+1, index+1)



if __name__ == "__main__":
    main()
