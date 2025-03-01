"""
Code adapted from: https://github.com/arunmallya/piggyback
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import gc
import torch.backends.cudnn as cudnn
import numpy as np
import random
import os
import tqdm

manualSeed = 42
DEFAULT_THRESHOLD = 5e-3

random.seed(manualSeed)
torch.manual_seed(manualSeed)
cudnn.benchmark = False
torch.backends.cudnn.enabled = False
# seed numpy
np.random.seed(manualSeed)

batch_size = 256
import gc


def calculate_parameters(model):
    params = sum([np.prod(p.size()) for p in model.parameters()])

    trainable_params = sum(
        [
            np.prod(p.size())
            for p in filter(lambda p: p.requires_grad, model.parameters())
        ]
    )
    untrainable_params = params - trainable_params
    template_params = 0
    mask_params = 0
    coefficients = 0
    feature_masks = 0
    new_conv_masks = 0
    for name, param in model.named_parameters():
        if "templates" in name:
            template_params += np.prod(param.size())
        if "template_masks" in name:
            mask_params += np.prod(param.size())

        if "output_masks" in name:
            feature_masks += np.prod(param.size())

        if "mask_real" in name:
            new_conv_masks += np.prod(param.size())

        if "coefficients" in name:
            coefficients += np.prod(param.size())

    print("* number of parameters: {}".format(params))
    print("* untrainable params: {}".format(untrainable_params))
    print("* trainable params: {}".format(trainable_params))

    print("* template params: {}".format(template_params))
    print("* mask params: {}".format(mask_params))
    print("* Shared convolutional feature mask params: {}".format(feature_masks))
    print("* convolutional mask params: {}".format(new_conv_masks))
    print("* coefficients: {}".format(coefficients))

    return (
        params,
        trainable_params,
        untrainable_params,
        template_params,
        mask_params,
        coefficients,
        feature_masks,
        new_conv_masks,
    )


import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
import os
import json
import numpy as np
from torchvision import datasets
import torch.nn as nn

from torchvision.transforms import autoaugment, transforms


# Write a base dataloader class for image classification
class ImageDataset(Dataset):
    def __init__(self):
        self.data_path = ""
        self.data_name = ""
        self.num_classes = 0
        self.train_transform = None
        self.train_csv_path = ""
        self.image_paths = []
        self.labels = []

    def get_num_classes(self):
        return self.num_classes

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        img = Image.open(img_path).convert("RGB")

        return img, label

    def __len__(self):
        return len(self.image_paths)

    @property
    def label_dict(self):
        return {i: self.class_map[i] for i in range(self.num_classes)}

    def __repr__(self):
        return f"ImageDataset({self.data_name}) with {self.__len__} instances"


class CARS(ImageDataset):
    def __init__(self):
        super().__init__()
        self.data_path = CARS_DATA
        self.data_name = "cars"
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.Normalize(
                    mean=[-0.0639, 0.0145, 0.2118], std=[1.2796, 1.3035, 1.3343]
                ),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[-0.0639, 0.0145, 0.2118], std=[1.2796, 1.3035, 1.3343]
                ),
            ]
        )
        self.train_csv_path = os.path.join(BASE_PATH, "cars.csv")
        self.image_paths = pd.read_csv(self.train_csv_path)["fname"].values
        self.labels = pd.read_csv(self.train_csv_path)["class"].values.tolist()
        self.num_classes = 196
        self.split = None
        # json file that contains the class names
        self.class_json = os.path.join(BASE_PATH, "CARS.json")
        self.class_map = json.load(open(self.class_json))


class AIRCRAFT(ImageDataset):
    def __init__(self):
        super().__init__()
        self.data_path = AIRCRAFT_DATA
        self.data_name = "aircraft"
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.Normalize(
                    mean=[-0.0266, 0.2407, 0.5663], std=[0.9745, 0.9684, 1.1040]
                ),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[-0.0266, 0.2407, 0.5663], std=[0.9745, 0.9684, 1.1040]
                ),
            ]
        )
        self.train_csv_path = os.path.join(BASE_PATH, "aircrafts.csv")
        self.image_paths = pd.read_csv(self.train_csv_path)["fname"].values
        self.labels = pd.read_csv(self.train_csv_path)["class"].values
        self.num_classes = 55
        self.split = None
        self.class_json = os.path.join(BASE_PATH, "AIRCRAFTS.json")
        self.class_map = json.load(open(self.class_json))


class FLOWERS(ImageDataset):
    def __init__(self):
        # (tensor([-0.2170, -0.3512, -0.5282]), tensor([1.1572, 0.9467, 0.9697]))

        super().__init__()
        self.data_path = FLOWERS_DATA
        self.data_name = "flowers"
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.Normalize(
                    mean=[0.5642, 0.7694, 0.8410], std=[0.2560, 0.2589, 0.2783]
                ),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.5642, 0.7694, 0.8410], std=[0.2560, 0.2589, 0.2783]
                ),
            ]
        )
        self.train_csv_path = os.path.join(BASE_PATH, "flowers.csv")
        self.image_paths = pd.read_csv(self.train_csv_path)["fname"].values
        self.labels = pd.read_csv(self.train_csv_path)["class"].values
        self.num_classes = 103  # not 0 indexed
        self.split = None
        self.class_json = os.path.join(BASE_PATH, "FLOWERS.json")
        self.class_map = json.load(open(self.class_json))


class SCENES(ImageDataset):
    def __init__(self):
        super().__init__()
        self.data_path = SCENES_DATA
        self.data_name = "scenes"
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.Normalize(
                    mean=[-0.0081, -0.1473, -0.1866], std=[1.1616, 1.1583, 1.1599]
                ),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[-0.0081, -0.1473, -0.1866], std=[1.1616, 1.1583, 1.1599]
                ),
            ]
        )
        self.train_csv_path = os.path.join(BASE_PATH, "scenes.csv")
        self.image_paths = pd.read_csv(self.train_csv_path)["fname"].values
        self.labels = pd.read_csv(self.train_csv_path)["class"].values
        self.num_classes = 67
        self.split = None
        self.class_json = os.path.join(BASE_PATH, "SCENES.json")
        self.class_map = json.load(open(self.class_json))


class CHARS(ImageDataset):
    def __init__(self):
        super().__init__()
        self.data_path = CHARS_DATA
        self.data_name = "chars"
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                # transforms.Normalize(mean=[1.4986, 1.6615, 1.8764], std=[1.6015, 1.6373, 1.6300])
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                # transforms.Normalize(mean=[1.4986, 1.6615, 1.8764], std=[1.6015, 1.6373, 1.6300])
            ]
        )
        self.train_csv_path = os.path.join(BASE_PATH, "chars.csv")
        self.image_paths = pd.read_csv(self.train_csv_path)["fname"].values
        self.labels = pd.read_csv(self.train_csv_path)["class"].values
        self.num_classes = 63  # not 0 indexed
        self.split = None
        self.class_json = os.path.join(BASE_PATH, "CHARS.json")
        self.class_map = json.load(open(self.class_json))


class BIRDS(ImageDataset):
    def __init__(
        self,
    ):
        super().__init__()
        self.data_path = BIRDS_DATA
        self.data_name = "birds"
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.Normalize(
                    mean=[0.0049, 0.1962, 0.1152], std=[1.0027, 1.0053, 1.1734]
                ),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                # transforms.Normalize(mean=[0.0049, 0.1962, 0.1152], std=[1.0027, 1.0053, 1.1734])
            ]
        )
        self.train_csv_path = os.path.join(BASE_PATH, "birds.csv")
        self.image_paths = pd.read_csv(self.train_csv_path)["fname"].values
        self.labels = pd.read_csv(self.train_csv_path)["class"].values
        self.num_classes = 201  # not 0 indexed
        self.split = None
        self.class_json = os.path.join(BASE_PATH, "BIRDS.json")
        self.class_map = json.load(open(self.class_json))


class ACTION(ImageDataset):
    def __init__(self):
        super().__init__()
        self.data_path = ACTION_DATA
        self.data_name = "actions"
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.train_csv_path = os.path.join(BASE_PATH, "action.csv")
        self.image_paths = pd.read_csv(self.train_csv_path)["fname"].values
        self.labels = pd.read_csv(self.train_csv_path)["class"].values
        self.num_classes = 20  # not 0 indexed
        self.split = None
        self.class_json = os.path.join(BASE_PATH, "ACTION.json")
        self.class_map = json.load(open(self.class_json))


class SVHN(ImageDataset):
    # TODO: ektu tricky beparshepar
    def __init__(self, split="train", transform=None):
        super().__init__()
        self.data_path = SVHN_DATA
        self.data_name = "svhn"
        self.task_id = 6  # Assign a unique task_id for SVHN
        self.split = split
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.dataset = datasets.SVHN(root=SVHN_DATA, split=split, download=True)
        self.num_classes = 10

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transform:
            img = self.transform(img)
        return img, label, self.task_id

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    images, labels, task_ids = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)
    task_ids = task_ids[0]
    return images, labels


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if isinstance(img, torch.Tensor):
            img = img.numpy().transpose(1, 2, 0)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)


def get_dataloaders(
    dataset_name, train_size=0.7, val_size=0.15, batch_size=32, mode="train"
):
    if dataset_name == "cars":
        dataset = CARS()
    elif dataset_name == "aircraft":
        dataset = AIRCRAFT()
    elif dataset_name == "flowers":
        dataset = FLOWERS()
    elif dataset_name == "scenes":
        dataset = SCENES()
    elif dataset_name == "chars":
        dataset = CHARS()
    elif dataset_name == "birds":
        dataset = BIRDS()
    elif dataset_name == "actions":
        dataset = ACTION()
    elif dataset_name == "cifar10":
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
            ]
        )
        cifar_train = datasets.CIFAR10(
            root=CIFAR_DATA, train=True, download=True, transform=train_transform
        )
        cifar_test = datasets.CIFAR10(
            root=CIFAR_DATA, train=False, download=True, transform=test_transform
        )

        train_size = int(train_size * len(cifar_train))
        val_size = int(val_size * len(cifar_train))
        test_size = len(cifar_train) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            cifar_train, [train_size, val_size, test_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        return train_loader, val_loader, test_loader, 10

    elif dataset_name == "cifar100":
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(*stats, inplace=True),
            ]
        )
        cifar_train = datasets.CIFAR100(
            root=CIFAR_DATA, train=True, download=True, transform=train_transform
        )
        cifar_test = datasets.CIFAR100(
            root=CIFAR_DATA, train=False, download=True, transform=transforms.ToTensor()
        )

        train_size = int(train_size * len(cifar_train))
        val_size = int(val_size * len(cifar_train))
        test_size = len(cifar_train) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            cifar_train, [train_size, val_size, test_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        return train_loader, val_loader, test_loader, 100

    elif dataset_name == "svhn":
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
            ]
        )
        svhn_train = SVHN(split="train", transform=train_transform)
        svhn_test = SVHN(split="test", transform=test_transform)

        train_size = int(train_size * len(svhn_train))
        val_size = int(val_size * len(svhn_train))
        test_size = len(svhn_train) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            svhn_train, [train_size, val_size, test_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        return train_loader, val_loader, test_loader, 10

    elif dataset_name == "imagenet":
        train_dataset = datasets.ImageFolder(
            IMAGENET_DATA,
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            ),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        test_path = IMAGENET_DATA.replace("train", "val")
        test_dataset = datasets.ImageFolder(
            test_path,
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            ),
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, None, test_loader, 1000

    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    # split the dataset into train, val, and test
    train_size = int(train_size * len(dataset))
    val_size = int(val_size * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(
        f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}"
    )
    # print(dataset.train_transform)
    print(dataset.test_transform)
    # Create transformed datasets for each split
    train_dataset = TransformedDataset(train_dataset, dataset.train_transform)
    val_dataset = TransformedDataset(val_dataset, dataset.test_transform)
    test_dataset = TransformedDataset(test_dataset, dataset.test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, dataset.get_num_classes()


MULT = 1
GEN_KERNEL = 3
num_cf = 2


class TemplateBank(nn.Module):
    def __init__(self, num_templates, in_planes, out_planes, kernel_size):
        super(TemplateBank, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.coefficient_shape = (num_templates, 1, 1, 1, 1)
        self.kernel_size = kernel_size
        templates = [
            torch.Tensor(out_planes, in_planes, kernel_size, kernel_size)
            for _ in range(num_templates)
        ]
        for i in range(num_templates):
            init.kaiming_normal_(templates[i])
        self.templates = nn.Parameter(
            torch.stack(templates)
        )  # this is what we will freeze later

    def forward(self, coefficients):
        weights = (self.templates * coefficients).sum(0)
        return weights

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + "num_templates="
            + str(self.coefficient_shape[0])
            + ", kernel_size="
            + str(self.kernel_size)
            + ")"
            + ", in_planes="
            + str(self.in_planes)
            + ", out_planes="
            + str(self.out_planes)
        )


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs, threshold):
        inputs = torch.tensor(inputs)  # Convert inputs to PyTorch tensor
        threshold = torch.tensor(threshold)  # Convert threshold to PyTorch tensor
        ctx.save_for_backward(inputs)
        return (inputs > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (inputs,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None


class MaskedConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False
    ):
        super(MaskedConv2d, self).__init__()
        mask_scale = 1e-2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.mask_real = nn.Parameter(
            torch.ones_like(self.conv.weight).uniform_(-1 * mask_scale, mask_scale)
            + DEFAULT_THRESHOLD,
            requires_grad=True,
        )
        self.threshold = nn.Parameter(torch.zeros(1))
        nn.init.uniform_(self.threshold, -1 * DEFAULT_THRESHOLD, DEFAULT_THRESHOLD)
        self.binarizer = Binarizer.apply

    def forward(self, x):
        mask = self.binarizer(self.mask_real, self.threshold)
        masked_weight = self.conv.weight * mask
        return F.conv2d(
            x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding
        )


class SConv2d(nn.Module):
    def __init__(
        self,
        bank,
        stride=1,
        padding=1,
        threshold=None,
        mask_scale=1e-2,
        scaling_factor=0.1,
    ):
        super(SConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.bank = bank
        self.coefficients = nn.ParameterList(
            [nn.Parameter(torch.zeros(bank.coefficient_shape)) for _ in range(num_cf)]
        )
        self.threshold_fn = Binarizer.apply
        self.threshold = nn.Parameter(
            torch.zeros(1), requires_grad=True
        )  # intiialize real-valued threshold
        nn.init.uniform_(self.threshold, -1 * DEFAULT_THRESHOLD, DEFAULT_THRESHOLD)

        # intiialize real-valued mask weights
        self.mask_real = nn.Parameter(
            torch.ones(bank.templates.shape[1], bank.templates.shape[1], 3, 3).uniform_(
                -1 * mask_scale, mask_scale
            )
            + DEFAULT_THRESHOLD,
            requires_grad=True,
        )

    def forward(self, input):
        params = torch.stack([self.bank(coeff) for coeff in self.coefficients]).mean(0)
        mask = self.threshold_fn(self.mask_real, self.threshold)
        weight_thresholded = params * mask
        return F.conv2d(
            input, weight_thresholded, stride=self.stride, padding=self.padding
        )


import copy


class CustomResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        bank1=None,
        bank2=None,
    ):
        super(CustomResidualBlock, self).__init__()
        self.bank1 = bank1
        self.bank2 = bank2

        # Ensure padding is always 1 for 3x3 convolutions
        if self.bank1 and self.bank2:
            self.conv1 = SConv2d(bank1, stride=stride, padding=1)
            self.conv2 = SConv2d(bank2, stride=1, padding=1)
        else:
            self.conv1 = MaskedConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.conv2 = MaskedConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Implement downsample as 1x1 convolution when needed
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                MaskedConv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, SConv2d):
                for coefficient in m.coefficients:
                    nn.init.orthogonal_(coefficient)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetTPB(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNetTPB, self).__init__()
        self.inplanes = 64
        self.layers = layers
        # self.conv1 = nn.Conv2d(
        # 3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        # )
        self.conv1 = MaskedConv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                # nn.Conv2d(
                #     self.inplanes, planes, kernel_size=1, stride=stride, bias=False
                # ),
                MaskedConv2d(
                    self.inplanes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        # Calculate parameters for remaining blocks
        params_per_conv = 9 * planes * planes
        params_per_template = 9 * planes * planes
        num_templates1 = max(
            1, int((blocks - 1) * params_per_conv / params_per_template)
        )
        num_templates2 = (
            num_templates1  # You could potentially use a different calculation here
        )

        print(
            f"Layer with {planes} planes, {blocks} blocks, using {num_templates1} templates for conv1 and {num_templates2} for conv2"
        )

        # Create separate TemplateBanks for conv1 and conv2
        tpbank1 = TemplateBank(num_templates1, planes, planes, GEN_KERNEL)
        tpbank2 = TemplateBank(num_templates2, planes, planes, GEN_KERNEL)

        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    bank1=tpbank1,
                    bank2=tpbank2,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ModResnet(nn.Module):
    def __init__(self, num_classes=10):
        super(ModResnet, self).__init__()
        pre_model = ResNetTPB(
            block=CustomResidualBlock,
            layers=[3, 4, 6, 3],
            num_classes=1000,
        )
        trained_weight_dict = torch.load(SHARED_WEIGHT)["state_dict"]
        new_state_dict = {}
        for k, v in trained_weight_dict.items():
            if "conv" in k and "weight" in k and ".conv." not in k:
                # This is for the main convolutions (conv1, conv2)
                new_k = k.replace(".weight", ".conv.weight")
                new_state_dict[new_k] = v

                # Initialize mask_real and threshold
                mask_k = k.replace(".weight", ".mask_real")
                threshold_k = k.replace(".weight", ".threshold")
                new_state_dict[mask_k] = torch.ones_like(v).uniform_(-0.01, 0.01) + 0.5
                new_state_dict[threshold_k] = torch.zeros(1)
            elif "downsample.0.weight" in k:
                # This is for the downsample layers
                new_k = k.replace(".weight", ".conv.weight")
                new_state_dict[new_k] = v

                # Initialize mask_real and threshold for downsample
                mask_k = k.replace(".weight", ".mask_real")
                threshold_k = k.replace(".weight", ".threshold")
                new_state_dict[mask_k] = torch.ones_like(v).uniform_(-0.01, 0.01) + 0.5
                new_state_dict[threshold_k] = torch.zeros(1)
            else:
                new_state_dict[k] = v
        print(pre_model.load_state_dict(new_state_dict, strict=False))
        resnet_embeddim = pre_model.fc.in_features
        pre_model.fc = nn.Linear(resnet_embeddim, num_classes, bias=True)
        pre_model.fc.weight.requires_grad = True
        pre_model.fc.bias.requires_grad = True
        # create shared feature generator
        self.shared = nn.Sequential()
        for name, module in pre_model.named_children():
            if name != "fc":
                self.shared.add_module(name, module)

        # set everything other than the mask_real parameter in the shared module to requires_grad = False
        for name, param in self.shared.named_parameters():
            if "mask_real" not in name or "coefficients" not in name:
                param.requires_grad = False
            else:
                print(name)
        self.classifier = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


import json
from torchvision.models import resnet34


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_data_loader = train_loader
        self.test_data_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()

    def eval(self):
        """Performs evaluation."""
        self.model.eval()
        error_meter = None
        device = "cpu"
        self.model = self.model.to(device)

        print("Performing eval...")
        for batch, label in tqdm.tqdm(self.test_data_loader, desc="Eval"):
            batch = batch.to(device)
            label = label.to(device)
            output = self.model(batch)

            # Init error meter.
            if error_meter is None:
                topk = [1]
                if output.size(1) > 5:
                    topk.append(5)
                error_meter = tnt.meter.ClassErrorMeter(topk=topk)
            error_meter.add(output.data, label)

        errors = error_meter.value()
        print("Error: " + ", ".join("@%s=%.2f" % t for t in zip(topk, errors)))

        self.model.train()

        return errors

    def do_batch(self, optimizer, batch, label):
        # IMPORTANT
        """Runs model for one batch."""
        self.model = self.model.to(device)
        batch = batch.to(device)
        label = label.to(device)

        # Set grads to 0.
        self.model.zero_grad()
        # Do forward-backward.
        output = self.model(batch)

        loss = self.criterion(output, label)
        loss.backward()

        # Set batchnorm grads to 0, if required.
        for module in self.model.shared.modules():
            if "BatchNorm" in str(type(module)):
                if module.weight.grad is not None:
                    module.weight.grad.data.fill_(0)
                if module.bias.grad is not None:
                    module.bias.grad.data.fill_(0)

        # Update params.
        optimizer.step()

    def do_epoch(self, epoch_idx, optimizer):
        """Trains model for one epoch."""

        for batch, label in tqdm.tqdm(
            self.train_data_loader, desc="Epoch: %d " % (epoch_idx)
        ):
            self.do_batch(optimizer, batch, label)

        print("Num 0ed out parameters:")
        total_zeroes = 0
        total_params = 0
        for idx, module in enumerate(self.model.shared.modules()):
            if "SConv2d" in str(type(module)) or "MaskedConv2d" in str(type(module)):
                module_threshold = module.threshold.item()
                num_zero = module.mask_real.data.lt(module_threshold).sum()
                total = module.mask_real.data.numel()
                print(idx, num_zero, total)
                total_zeroes += num_zero
                total_params += total
        print("Current epoch: ", epoch_idx)
        print(f"Total zeroed out parameters: {total_zeroes}")
        print(f"Total parameters: {total_params}")
        print(
            f"Percentage of zeroed out parameters: {(total_zeroes/total_params) * 100}%"
        )

        print("-" * 20)

    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        # Prepare the ckpt.
        ckpt = {
            "epoch": epoch,
            "accuracy": best_accuracy,
            "errors": errors,
            "model": self.model,
        }

        # Save to file.
        torch.save(ckpt, savename)

    def train(self, epochs, optimizer, save=True, savename="", best_accuracy=0):
        """Performs training."""
        best_accuracy = best_accuracy
        error_history = []
        self.model = self.model.to(device)

        self.eval()
        val_accuracies = []
        for idx in tqdm.tqdm(range(epochs), desc="Epoch"):
            epoch_idx = idx + 1
            print("Epoch: %d" % (epoch_idx))

            optimizer.update_lr(epoch_idx)
            self.model.train()
            self.do_epoch(epoch_idx, optimizer)
            errors = self.eval()
            error_history.append(errors)
            accuracy = 100 - errors[0]  # Top-1 accuracy.
            val_accuracies.append(accuracy)
            print(f"Epoch {epoch_idx} Val Accuracy: {accuracy}%")
            # Save best model, if required.
            if save and accuracy > best_accuracy:
                print(
                    "Best model so far, Accuracy: %0.2f%% -> %0.2f%%"
                    % (best_accuracy, accuracy)
                )
                best_accuracy = accuracy
                self.save_model(epoch_idx, best_accuracy, errors, savename)

        # Make sure masking didn't change any weights.
        print("Finished finetuning...")
        print(
            "Best error/accuracy: %0.2f%%, %0.2f%%"
            % (100 - best_accuracy, best_accuracy)
        )
        print(f"Average accuracy: {sum(val_accuracies) / len(val_accuracies)}")
        print("-" * 16)

        print("Checking for zeroed out parameters...")
        total_zeroes = 0
        total_params = 0
        for idx, module in enumerate(self.model.shared.modules()):
            if "SConv2d" in str(type(module)) or "MaskedConv2d" in str(type(module)):
                num_zero = module.mask_real.data.lt(5e-3).sum()
                total = module.mask_real.data.numel()
                total_zeroes += num_zero
                total_params += total
                print(idx, num_zero, total)

        print(f"Total zeroed out parameters: {total_zeroes}")
        print(f"Total parameters: {total_params}")
        print(
            f"Percentage of zeroed out parameters: {(total_zeroes/total_params) * 100}%"
        )
        print("-" * 16)


def step_lr(epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
    """Handles step decay of learning rate."""
    factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
    new_lr = base_lr * factor
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    print("Set lr to ", new_lr)
    return optimizer


class Optimizers(object):
    """Handles a list of optimizers."""

    def __init__(self):
        self.optimizers = []
        self.lrs = []
        self.decay_every = []
        self.lr_decay_factor = 0.1

    def add(self, optimizer, lr, decay_every):
        """Adds optimizer to list."""
        self.optimizers.append(optimizer)
        self.lrs.append(lr)
        self.decay_every.append(decay_every)

    def step(self):
        """Makes all optimizers update their params."""
        for optimizer in self.optimizers:
            optimizer.step()

    def update_lr(self, epoch_idx):
        """Update learning rate of every optimizer."""
        for optimizer, init_lr, decay_every in zip(
            self.optimizers, self.lrs, self.decay_every
        ):
            optimizer = step_lr(
                epoch_idx, init_lr, decay_every, self.lr_decay_factor, optimizer
            )


# TASK_NAME = ["flowers", "aircraft", "scenes"]
# num_classes = [103, 56, 67]

# TASK_NAME = ["cifar100", "cifar10", "birds"]
# num_classes = [100, 10, 201]


classes_per_task = num_classes
TASK_LIST = TASK_NAME
print(f"Tasks: {TASK_LIST}")
print(f"Num classes: {classes_per_task}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# load dataset pickle

lr_mask = 1e-4
lr_mask_decay_every = 30
lr_classifier = 3e-2
lr_classifier_decay_every = 33
lr_cf = 5e-4
lr_cf_decay_every = 33
for num_classes, task_name in zip(classes_per_task, TASK_LIST):
    print(
        f"**********Start Training task: {task_name} with {num_classes} classes **********"
    )
    # clear memory
    torch.cuda.empty_cache()
    gc.collect()
    piggy = ModResnet(num_classes=num_classes).to(device)
    print(piggy.classifier)

    if task_name == "cifar100" or task_name == "cifar10":
        print("Using CIFAR10/100 dataset")
        train_loader, _, test_loader, num_class = get_dataloaders(
            task_name, train_size=0.8, val_size=0.0, batch_size=256, mode="train"
        )
    else:
        train_loader = dataloader_dict[task_name]["train"]
        test_loader = dataloader_dict[task_name]["test"]
    manager = Manager(piggy, train_loader, test_loader)
    cf_params = []
    for n, p in piggy.named_parameters():
        if "mask_real" in n or "classifier" in n or "head" in n:
            p.requires_grad = True
        if "coefficients" in n:
            p.requires_grad = True
            cf_params.append(p)
        if "threshold" in n:
            p.requires_grad = True
        if p.requires_grad:
            print(f"> TRAINABLE:  {n} {p.numel()}")
    calculate_parameters(piggy)
    # Two optimizers, one for masks and one for classifier.
    optimizer_masks = torch.optim.Adam(  # IMPORTANT
        piggy.shared.parameters(), lr=lr_mask
    )
    optimizer_classifier = torch.optim.Adam(
        piggy.classifier.parameters(), lr=lr_classifier
    )
    optimizer_cf = torch.optim.Adam(cf_params, lr=lr_cf)

    optimizers = Optimizers()
    optimizers.add(optimizer_masks, lr_mask, lr_mask_decay_every)
    optimizers.add(optimizer_classifier, lr_classifier, lr_classifier_decay_every)
    optimizers.add(optimizer_cf, lr_cf, lr_cf_decay_every)
    print("Optimizers: ", optimizers.optimizers)

    total_params = sum(p.numel() for p in piggy.parameters())
    fc_params = sum(p.numel() for p in piggy.classifier.parameters())
    trainable_params = sum(
        p.numel() for p in piggy.shared.parameters() if p.requires_grad
    ) + sum(p.numel() for p in piggy.classifier.parameters() if p.requires_grad)

    print("Some double checking before training starts")
    total_zeroes = 0
    total_params = 0
    for idx, module in enumerate(piggy.shared.modules()):
        if "SConv2d" in str(type(module)) or "MaskedConv2d" in str(type(module)):
            num_zero = module.mask_real.data.lt(5e-3).sum()
            total = module.mask_real.data.numel()
            total_zeroes += num_zero
            total_params += total

    print(f"Total zeroed out parameters: {total_zeroes}")
    print(f"Total parameters: {total_params}")
    print(f"Percentage of zeroed out parameters: {(total_zeroes/total_params) * 100}%")

    manager.train(100, optimizers, save=True, savename=f"piggytpb_full_{task_name}")
    # check the threshold values
    for idx, module in enumerate(piggy.shared.modules()):
        if "SConv2d" in str(type(module)) or "MaskedConv2d" in str(type(module)):
            print(f"Module {idx} has threshold: {module.threshold}")
    del piggy, train_loader, test_loader, manager, optimizers
    torch.cuda.empty_cache()
    gc.collect()
