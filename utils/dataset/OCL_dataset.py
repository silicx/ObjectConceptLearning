from functools import lru_cache
import sys
import pickle

assert sys.version.startswith('3'), "Python version >= 3.8"
from PIL import Image
import numpy as np
import os, json
import logging

import torch, torchvision
import torch.utils.data as tdata

from . import data_utils

class ImageDataset(tdata.Dataset):
    def __init__(self, phase):
        super(ImageDataset, self).__init__()
        self.logger = logging.getLogger(f"Dataset OCL")
        self.phase = phase
        self.transform = data_utils.imagenet_transform(phase)

        # reading annotation pickle
        pkl_dir = "data/resources"

        def load_pickle_and_assign_split(split):
            pkl_path = os.path.join(pkl_dir, f"OCL_annot_{split}.pkl")
            with open(pkl_path, 'rb') as fp:
                pkl = pickle.load(fp)
            for x in pkl:
                x['split'] = split
            return pkl

        if phase == "train":
            self.pkl_data = load_pickle_and_assign_split("train")
        else:
            self.pkl_data = load_pickle_and_assign_split("val") + \
                    load_pickle_and_assign_split("test")

        self.instance_indices = [(i, j) for i, img in enumerate(self.pkl_data) for j in range(len(img['objects']))]
        self.logger.info(f"{len(self.instance_indices)} instances")

        # construct attr/aff/obj list/matrix
        def load_class_json(name):
            with open(os.path.join(pkl_dir, f"OCL_class_{name}.json"), "r") as fp:
                return json.load(fp)
            
        self.attrs = load_class_json("attribute")
        self.objs = load_class_json("object")
        self.affs = load_class_json("affordance")
        self.obj2id = {x: i for i, x in enumerate(self.objs)}

        with open('data/resources/category_aff_matrix.json', "r") as fp:
            aff_matrix_file = json.load(fp)
            assert self.objs == aff_matrix_file['objs']
            self.aff_matrix = np.array(aff_matrix_file["aff_matrix"])

        self.num_aff = self.aff_matrix.shape[1]
        self.num_attr = len(self.attrs)
        self.num_obj = len(self.objs)

        logging.info("#obj %d, #attr %d, #aff %d" % (
            self.num_obj, self.num_attr, self.num_aff))


    def __len__(self):
        return len(self.pkl_data)

    def __getitem__(self, index):
        info = self.pkl_data[index]
        objects = info["objects"]
        file_name = info["name"]
        file_path = os.path.join("data", file_name)
        image = Image.open(file_path).convert('RGB')
        image_width, image_height = image.size
        
        attr = []
        aff = []
        gt_box = []
        for obj in objects:
            a = torch.zeros([self.num_attr]).float()
            a[obj['attr']] = 1
            attr.append(a)

            b = torch.zeros([self.num_aff]).float()
            b[obj['aff']] = 1
            aff.append(b)

            x = obj.get('bbox', [0, 0, image_width, image_height])
            gt_box.append(x)

        attr = torch.stack(attr, 0)
        aff = torch.stack(aff, 0)


        if max(image_width, image_height) > 1800:
            image_width, image_height = image_width // 2, image_height // 2
            resize_trans = torchvision.transforms.Resize((image_width, image_height))
            image = resize_trans(image)
            gt_box = [(x[0] // 2, x[1] // 2, x[2] // 2, x[3] // 2) for x in gt_box]
        image = self.transform(image)


        sample = {
            "image": image,
            "file_name": file_name,
            "gt_bbox": np.array(gt_box, dtype=np.float32),
            'gt_aff': aff,
            "gt_attr": attr,
            'gt_obj_id': np.array([self.obj2id[obj['obj']] for obj in objects]),
            'val_mask': np.array([info['split'] == 'val' for _ in objects])
        }

        sample["main_bbox"] = sample["gt_bbox"]

        return sample



class FeatureDataset(ImageDataset):
    def __init__(self, phase, feature_dir):
        super(FeatureDataset, self).__init__(phase)

        # reading feature .t7
        feature_path = os.path.join(feature_dir, f"{phase}.t7")
        logging.info("reading " + feature_path)

        # load features into memory
        self.features_list, self.feature_dim = data_utils.features_loader(feature_path, self.pkl_data)

    def __len__(self):
        return len(self.instance_indices)

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        imgId, instId = self.instance_indices[index]
        img = self.pkl_data[imgId]
        obj = img['objects'][instId]

        feature = self.features_list[imgId][instId, ...]

        obj_id = self.obj2id[obj['obj']]

        attr = torch.zeros([self.num_attr]).float()
        attr[obj['attr']] = 1

        aff = torch.zeros([self.num_aff]).float()
        aff[obj['aff']] = 1
        

        sample = {
            "image": feature,
            "gt_attr": attr,
            'gt_obj_id': np.array(obj_id, dtype=int),
            'gt_aff': aff,
        }

        # add val/test mask
        if 'split' in img:
            sample['val_mask'] = np.array(img['split'] == "val")
        sample['gt_causal'] = np.array(obj['causal'],dtype=int) if 'causal' in obj else []

        return sample