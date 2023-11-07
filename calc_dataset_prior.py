import os, logging, tqdm, argparse, json
import torch
import os.path as osp
import numpy as np

from utils import dataset


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="OCKB")
parser.add_argument("--feature_dir", type=str, required=True)
args = parser.parse_args()


logging.info("Loading dataset")


train_loader = dataset.get_dataloader(
    'train', "feature", args.feature_dir,
    batchsize=1, shuffle=False)
dataset = train_loader.dataset

num_sample = 0
num_pos_obj  = np.zeros([dataset.num_obj], dtype=int)
num_pos_attr = np.zeros([dataset.num_attr], dtype=int)
num_pos_aff  = np.zeros([dataset.num_aff], dtype=int)

feat_of_obj  = [[] for _ in dataset.objs]
feat_of_attr = [[] for _ in range(dataset.num_attr)]
feat_of_aff  = [[] for _ in range(dataset.num_aff)]


for batch in tqdm.tqdm(train_loader):
    num_sample += 1
    gt_obj  = batch['gt_obj_id'][0].item()
    gt_attr = (batch['gt_attr'][0].numpy()>0.5).astype(int)
    gt_aff  = (batch['gt_aff'][0].numpy()>0.5).astype(int)
    feature = batch['image'][0]

    num_pos_obj[gt_obj] += 1
    num_pos_attr += gt_attr
    num_pos_aff += gt_aff

    feat_of_obj[gt_obj].append(feature)
    for i,x in enumerate(gt_attr.tolist()):
        if x==1:
            feat_of_attr[i].append(feature)
    for i,x in enumerate(gt_aff.tolist()):
        if x==1:
            feat_of_aff[i].append(feature)


def calc_pos_weight(n_pos, max_weight=50):
    wgt = float(num_sample) / (n_pos+1e-6)
    wgt = np.minimum(wgt, max_weight)
    return wgt

pos_weight_attr = calc_pos_weight(num_pos_attr)
pos_weight_aff  = calc_pos_weight(num_pos_aff)


num_pos_obj[num_pos_obj==0]   = 1
num_pos_attr[num_pos_attr==0] = 1
num_pos_aff[num_pos_aff==0]   = 1
pos_obj_freq  = num_pos_obj / float(num_sample)
pos_attr_freq = num_pos_attr / float(num_sample)
pos_aff_freq  = num_pos_aff / float(num_sample)

obj_weight  = -np.log(pos_obj_freq)
attr_weight = -np.log(pos_attr_freq)
aff_weight  = -np.log(pos_aff_freq)
obj_weight  = obj_weight / np.mean(obj_weight)  # mean: not a bug, let sum(weight)=len(weight)
attr_weight = attr_weight / np.mean(attr_weight)
aff_weight  = aff_weight / np.mean(aff_weight)

with open(f"data/resources/OCL_weight.json", "w") as fp:
    json.dump({
        "obj_weight": obj_weight.tolist(),
        "attr_weight":  attr_weight.tolist(),
        "aff_weight":  aff_weight.tolist(),
    }, fp)

with open(f"data/resources/OCL_bce_pos_weight.json", "w") as fp:
    json.dump({
        "attr": pos_weight_attr.tolist(),
        "aff":  pos_weight_aff.tolist(),
    }, fp)

zero_feature = torch.zeros([dataset.feature_dim], dtype=torch.float32)

feat_of_obj = [
    sum(x,0)/len(x) if len(x)>0 else zero_feature
    for x in feat_of_obj]
feat_of_obj = torch.stack(feat_of_obj, 0)

feat_of_attr = [
    sum(x,0)/len(x) if len(x)>0 else zero_feature
    for x in feat_of_attr]
feat_of_attr = torch.stack(feat_of_attr, 0)

feat_of_aff = [
    sum(x,0)/len(x) if len(x)>0 else zero_feature
    for x in feat_of_aff]
feat_of_aff = torch.stack(feat_of_aff, 0)


torch.save({
    "freqency": torch.from_numpy(pos_obj_freq).float(),
    "mean_obj_features": feat_of_obj,
    "mean_attr_features": feat_of_attr,
    "mean_aff_features": feat_of_aff,
}, osp.join(args.feature_dir, "obj_prior.t7"))
