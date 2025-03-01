"""
AdaptFormer code adapted from: https://github.com/ShoufaChen/AdaptFormer?utm_source=catalyzex.com
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
FACTORS = 6  # number of groups
TEMPLATES = (
    2  # number of templates per bank, corresponds to number of layers in a group
)
MULT = 1  # optional multiplier for the number of coefficients set
num_cf = 2  # number of coefficients sets per target module


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super(MLP, self).__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else in_features
        )
        linear_layer = nn.Linear if not use_conv else nn.Conv2d

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act_fn = act_layer()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()
        # self.norm = norm_layer(out_features) if norm_layer is not None else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.normal_(self.fc1.bias, std=1e-6)
            nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        # print("MLP input: ", x.shape)
        x = self.fc1(x)
        # print("MLP fc1: ", x.shape)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # print("MLP fc2: ", x.shape)
        x = self.dropout(x)
        # x = self.norm(x)
        return x


class AttTemplateBank(nn.Module):
    def __init__(self, num_templates, in_dim, num_patches):
        super().__init__()
        self.num_templates = num_templates
        self.in_dim = in_dim
        self.num_patches = num_patches + 1
        self.num_templates = num_templates
        templates = [
            torch.Tensor(in_dim * 3, in_dim) for _ in range(self.num_templates)
        ]
        for i in range(self.num_templates):
            nn.init.kaiming_normal_(templates[i], mode="fan_out", nonlinearity="relu")
        self.templates = nn.Parameter(torch.stack(templates, dim=0))
        self.coefficient_shape = (self.num_templates, 1, 1)

    def forward(self, coefficients):
        params = self.templates * coefficients
        summed_params = torch.sum(params, dim=0)
        return summed_params


class SharedAttention(nn.Module):
    # TARGET MODULE
    def __init__(self, template_bank, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.template_bank = template_bank
        self.num_heads = num_heads
        self.in_dim = template_bank.in_dim
        self.head_dim = self.in_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.in_dim, self.in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_bias = nn.Parameter(torch.zeros(3 * self.in_dim))
        self.qkv_coefficients = nn.ParameterList(
            nn.Parameter(torch.zeros(self.template_bank.coefficient_shape))
            for _ in range(num_cf)
        )

        for coeff in self.qkv_coefficients:
            nn.init.orthogonal_(coeff)

    def forward(self, x):
        B, N, C = x.shape
        qkv_weights = torch.stack(
            [self.template_bank(coeff) for coeff in self.qkv_coefficients], dim=0
        ).mean(dim=0)
        qkv = F.linear(x, qkv_weights, bias=self.qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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
        self.attn = SharedAttention(
            bank1, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop
        )
        # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = nn.Identity()
        self.ls2 = nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=drop,
        )

    def forward(self, x):
        # print("Block input: ", x.shape)
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

        self.template_banks = nn.ModuleList(
            [
                AttTemplateBank(self.num_templates, embed_dim, num_patches)
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
            bank = self.template_banks[group_idx]
            # print("Current group: ", group_idx)
            # print("Current template: ", current_template)
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
                    bank1=bank,
                    bank2=None,
                )
            )
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


class _LoRA_Block(nn.Module):
    def __init__(
        self,
        block: MLP,
        linear_a_1: nn.Module,
        linear_b_1: nn.Module,
        r: int,
        alpha: int,
    ):
        super().__init__()
        self.block = block
        self.down_proj = linear_a_1
        self.up_proj = linear_b_1
        self.lora_dim = r
        self.alpha = alpha
        self.norm1 = block.norm1
        self.ls1 = block.ls1
        self.norm2 = block.norm2
        self.ls2 = block.ls2
        self.drop_path = block.drop_path
        self.scale = self.alpha // self.lora_dim
        self.attn = block.attn
        self.act = block.mlp.act_fn
        self.drop = block.mlp.dropout
        self.fc1 = block.mlp.fc1
        self.fc2 = block.mlp.fc2

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))

        # self.adaptmlp()
        down = self.down_proj(x)
        down = F.relu(down)  # TODO
        up = self.up_proj(down)
        adapt_x = up * self.scale  # adapt_x
        ##############################
        residual = x
        x = self.norm2(x)
        x = self.fc1(x)  # self.fc1
        x = self.drop(self.act(x))
        x = self.fc2(x)  # self.fc2
        x = self.drop_path(self.drop(x))
        x = x + adapt_x
        return residual + x


class AdaptMLP(nn.Module):
    def __init__(self, vit_model, r=2, alpha=4):
        super(AdaptMLP, self).__init__()
        self.alpha = alpha
        self.r = r
        self.lora_vit = vit_model
        self.embed_dim = vit_model.embed_dim
        self.num_groups = 12  # TODO
        assert r > 0
        assert alpha > 1
        print("Num groups: ", self.num_groups)
        print(f"Rank: {r}, Alpha: {alpha}")
        lora_As = nn.ModuleList(
            nn.Linear(self.embed_dim, self.r) for _ in range(self.num_groups)
        )
        lora_Bs = nn.ModuleList(
            nn.Linear(self.r, self.embed_dim) for _ in range(self.num_groups)
        )
        for param in self.lora_vit.parameters():
            param.requires_grad = False
        for i, block in enumerate(vit_model.blocks):
            group_idx = i
            lora_A = lora_As[group_idx]
            lora_B = lora_Bs[group_idx]

            nn.init.kaiming_normal_(lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(lora_B.weight)
            nn.init.zeros_(lora_A.bias)
            nn.init.zeros_(lora_B.bias)

            lora_block = _LoRA_Block(
                block=block,
                linear_a_1=lora_A,
                linear_b_1=lora_B,
                r=self.r,
                alpha=self.alpha,
            )
            self.lora_vit.blocks[i] = lora_block

    def forward(self, x):
        return self.lora_vit(x)


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
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.size())
    model = AdaptMLP(model)
    y = model(x)
    print(model)
    print(y.size())
