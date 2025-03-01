"""
Piggyback code adapted from: https://github.com/arunmallya/piggyback
"""

import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import math

manualSeed = 42
DEFAULT_THRESHOLD = 5e-3

random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
np.random.seed(manualSeed)
cudnn.benchmark = False
torch.backends.cudnn.enabled = False
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
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
            nn.init.kaiming_normal_(templates[i])
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


class SConv2d(nn.Module):
    # TARGET MODULE
    def __init__(self, bank, stride=1, padding=1):
        super(SConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.bank = bank
        self.num_templates = bank.coefficient_shape[0]

        self.coefficients = nn.ParameterList(
            [nn.Parameter(torch.zeros(bank.coefficient_shape)) for _ in range(num_cf)]
        )

    def forward(self, input):
        param_list = []
        for i in range(len(self.coefficients)):
            params = self.bank(self.coefficients[i])
            param_list.append(params)

        final_params = torch.stack(param_list).mean(0)
        return F.conv2d(input, final_params, stride=self.stride, padding=self.padding)


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
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.conv2 = nn.Conv2d(
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
                nn.Conv2d(
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
        self.conv1 = nn.Conv2d(
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
                nn.Conv2d(
                    self.inplanes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        # DYNAMICALLY CALCULATE THE NUMBER OF TEMPLATES TO USE FOR EACH RESIDUAL BLOCK
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


SHARED_WEIGHT = "<RECONSTRUCTEDWEIGHT.pt>"


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


def test():
    net = ModResnet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    print(net)
