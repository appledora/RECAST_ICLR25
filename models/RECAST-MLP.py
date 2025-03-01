import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random


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
FACTORS = 6  # number of groups
TEMPLATES = (
    2  # number of templates per bank, corresponds to number of layers in a group
)
MULT = 1  # optional multiplier for the number of coefficients set
num_cf = 2  # number of coefficients sets per target module


class MLPTemplateBank(nn.Module):
    def __init__(self, num_templates, in_features, out_features):
        super(MLPTemplateBank, self).__init__()
        self.num_templates = num_templates
        self.coefficient_shape = (num_templates, 1, 1)
        templates = [
            torch.Tensor(out_features, in_features) for _ in range(num_templates)
        ]
        for i in range(num_templates):
            nn.init.kaiming_normal_(templates[i])
        self.templates = nn.Parameter(torch.stack(templates))

    def forward(self, coefficients):
        params = self.templates * coefficients
        summed_params = torch.sum(params, dim=0)
        return summed_params

    def __repr__(self):
        return f"MLPTemplateBank(num_templates={self.templates.shape[0]}, in_features={self.templates.shape[1]}, out_features={self.templates.shape[2]}, coefficients={self.coefficient_shape})"


class SharedMLP(nn.Module):
    def __init__(
        self, bank1, bank2, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=0.0
    ):
        super(SharedMLP, self).__init__()
        self.bank1 = None
        self.bank2 = None

        if bank1 != None and bank2 != None:
            self.bank1 = bank1
            self.bank2 = bank2
            self.coefficients1 = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(bank1.coefficient_shape), requires_grad=True
                    )
                    for _ in range(num_cf)
                ]
            )
            self.coefficients2 = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(bank2.coefficient_shape), requires_grad=True
                    )
                    for _ in range(num_cf)
                ]
            )
            self.bias1 = nn.Parameter(torch.zeros(bank1.templates.shape[1]))
            self.bias2 = nn.Parameter(torch.zeros(bank2.templates.shape[1]))

        self.act = act_layer()
        self.norm = nn.Identity()
        self.drop = nn.Dropout(drop)
        self.init_weights()

    def init_weights(self):
        if self.bank1 != None:
            for cf in self.coefficients1:
                nn.init.orthogonal_(cf)
        if self.bank2 != None:
            for cf in self.coefficients2:
                nn.init.orthogonal_(cf)

    def forward(self, x):
        if self.bank1 != None:
            weight1 = []
            for c in self.coefficients1:
                w = self.bank1(c)
                weight1.append(w)
            weights1 = torch.stack(weight1).mean(0)
        if self.bank2 != None:
            weight2 = []
            for c in self.coefficients2:
                w = self.bank2(c)
                weight2.append(w)
            weights2 = torch.stack(weight2).mean(0)

        x = F.linear(x, weights1, self.bias1)
        x = self.act(x)
        x = self.norm(x)
        x = F.linear(x, weights2, self.bias2)
        x = self.drop(x)
        return x


# original timm module for vision transformer
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)  # attention proba
        attn = self.attn_drop(attn)
        x = attn @ v  # attention output
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        bank1=None,
        bank2=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = nn.Identity()
        self.ls2 = nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SharedMLP(
            bank1, bank2, act_layer=act_layer, norm_layer=norm_layer, drop=drop
        )

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = nn.Flatten(start_dim=2, end_dim=3)(x).permute(0, 2, 1)
        x = self.norm(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.img_size = img_size
        self.dim = embed_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_features = self.embed_dim
        self.num_prefix_tokens = 1
        self.num_patches = (img_size // patch_size) ** 2
        self.num_prefix_tokens = 1
        self.has_class_token = True
        self.cls_token = nn.Parameter(torch.ones(1, 1, self.embed_dim))
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = (self.img_size // self.patch_size) ** 2
        print("Num patches: ", num_patches)
        embed_len = num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(
            torch.ones(1, num_patches + self.num_prefix_tokens, embed_dim) * 0.02,
            requires_grad=True,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_drop = nn.Identity()
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)

        self.num_groups = FACTORS
        self.num_layers_in_group = (
            depth // self.num_groups
        )  # how many consective encoder layers share the same template bank
        print("Num layers in group: ", self.num_layers_in_group)
        self.num_templates = TEMPLATES
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.template_banks1 = nn.ModuleList(
            [
                MLPTemplateBank(self.num_templates, embed_dim, mlp_hidden_dim)
                for _ in range(self.num_groups)
            ]
        )
        self.template_banks2 = nn.ModuleList(
            [
                MLPTemplateBank(self.num_templates, mlp_hidden_dim, embed_dim)
                for _ in range(self.num_groups)
            ]
        )
        self.depth = depth

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, self.num_groups)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        for i in range(depth):
            group_idx = i // self.num_layers_in_group
            print(group_idx)
            bank1 = self.template_banks1[group_idx]
            bank2 = self.template_banks2[group_idx]
            self.blocks.append(
                Block(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[group_idx],
                    norm_layer=norm_layer,
                    bank1=bank1,
                    bank2=bank2,
                )
            )
        print(f"Num blocks: {len(self.blocks)}")
        self.norm = norm_layer(self.embed_dim)
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _pos_embed(self, x):
        to_cat = []
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        x = torch.cat(to_cat + [x], dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_head(self, x):
        x = x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        head_output = self.forward_head(features)
        return head_output


def test():
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    )
    print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
