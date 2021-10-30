import torch
import torch.nn as nn

# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py


class DropPath(nn.Module):

    def __init__(self, drop_prob=0., training=False):
        super().__init__()
        self.drop_prob = drop_prob
        self.training = training
        

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0]) + (1,) * (x.ndim - 1)  # broadcast
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binary mask
        output = x.div(keep_prob) * random_tensor
        return output

class MultiHeadAttention(nn.Module):
    """
    https://blog.csdn.net/qq_39478403/article/details/118704747
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
 
        self.num_heads = num_heads
        head_dim = dim // num_heads
 
        self.scale = qk_scale or head_dim ** -0.5
 
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
 
        # dropout
        self.proj_drop = nn.Dropout(proj_drop)
 
    def forward(self, x):
        # B: batch size; N: number of patch(word/token); C: patch(word/token) embeding
        B, N, C = x.shape
        # (B, N, 3, H, E) -> (3, B, H, N, E)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # (B, H, N, E) @ (B, H, E, N) -> (B, H, N, N) / attention of each head
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # normalization
        attn = attn.softmax(dim=-1)
        # dropout
        attn = self.attn_drop(attn)
        # (B, H, N, N) @ (B, H, N, E) -> (B, H, N, E) -> (B, N, H, E) -> (B, N, HxE)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # projection / (B, N, HxE) -> (B, N, HxE)
        x = self.proj(x)
        # dropout
        x = self.proj_drop(x)
 
        return x



class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
 
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
 
        return x


# Transformer Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
 
        # 后接于 MHA 的 Layer Norm
        self.norm1 = norm_layer(dim)
        # MHA
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
 
        # 后接于 MLP 的 Layer Norm
        self.norm2 = norm_layer(dim)
        # 隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP
        self.mlp = MultiLayerPerceptron(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
 
    def forward(self, x):
        # MHA + Add & Layer Norm
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # MLP + Add & Layer Norm
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x