import torch
import torchvision.transforms as transforms


def imagenet_transform(phase: str):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase=='train':
        transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
    elif phase in ['test', 'val', 'valtest']:
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
    else:
        raise

    return transform


def features_loader(feature_path, pkl_data, ignore_missing=False):
    features_list = []
    t7file = torch.load(feature_path)
    t7file = dict(zip(t7file["file_names"], t7file["features"]))
    feature_dim = None
    for img in pkl_data:
        k = img["name"]
        if k in t7file:
            features_list.append( t7file[k] )
            if feature_dim is None:
                feature_dim = t7file[k].size(1)
        else:
            if ignore_missing:
                features_list.append( [] )
            else:
                raise KeyError(k)
    return features_list, feature_dim
