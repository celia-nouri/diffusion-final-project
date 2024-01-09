import torch
import torch.nn as nn
import torch.nn.functional as F


def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    """
    Self attention module using multi-head attention
    """
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, pooling_type="max"):
        super().__init__()
        pooling_layer = nn.MaxPool2d(2)
        if pooling_type == "avg":
            pooling_layer = nn.AvgPool2d(2)
        self.pool_conv = nn.Sequential(
            pooling_layer,
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.pool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # TODO(celia): add option to use use_scale_shift_norm
        return x + emb


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Encoder(nn.Module):
    def __init__(self, input_channels=3, conv_channels=64):
        super().__init__()
        self.conv = DoubleConv(input_channels, conv_channels)
        self.downsample_1 = DownSample(conv_channels, conv_channels*2)
        self.attention_1 = SelfAttention(conv_channels*2)
        self.downsample_2 = DownSample(conv_channels*2, conv_channels*4)
        self.attention_2 = SelfAttention(conv_channels*4)
        self.downsample_3 = DownSample(conv_channels*4, conv_channels*4)
        self.attention_3 = SelfAttention(conv_channels*4)

    def forward(self, x, t):
        # some preprocessing on the image
        x1 = self.conv(x)
        x2 = self.downsample_1(x1, t)
        x2 = self.attention_1(x2)
        x3 = self.downsample_2(x2, t)
        x3 = self.attention_2(x3)
        x4 = self.downsample_3(x3, t)
        x4 = self.attention_3(x4)
        return x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self, input_channels, output_channels=3):
        super().__init__()
        self.upsample_1 = UpSample(input_channels*2, input_channels // 2)  # input_channels should be 256
        self.attention_1 = SelfAttention(input_channels // 2)
        self.upsample_2 = UpSample(input_channels, input_channels // 4)
        self.attention_2 = SelfAttention(input_channels // 4)
        self.upsample_3 = UpSample(input_channels // 2, input_channels // 4)
        self.attention_3 = SelfAttention(input_channels // 4)
        self.conv = nn.Conv2d(input_channels // 4, output_channels, kernel_size=1)

    def forward(self, x1, x2, x3, x4, t):
        x = self.upsample_1(x4, x3, t)
        x = self.attention_1(x)
        x = self.upsample_2(x, x2, t)
        x = self.attention_2(x)
        x = self.upsample_3(x, x1, t)
        x = self.attention_3(x)
        return self.conv(x)


class Bottleneck(nn.Module):
    def __init__(self, input_channels, equal_dim_conv=True):
        super().__init__()
        self.equal_dim_conv = equal_dim_conv
        self.deep_layer = nn.Sequential(
            DoubleConv(input_channels, input_channels*2),
            DoubleConv(input_channels*2, input_channels*2),
            DoubleConv(input_channels*2, input_channels),
        )
        self.equal_layer = nn.Sequential(
            DoubleConv(input_channels, input_channels),
            DoubleConv(input_channels, input_channels),
        )

    def forward(self, x):
        if self.equal_dim_conv:
            return self.equal_layer(x)
        return self.deep_layer(x)


class UNetModule(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, time_dim=256, equal_dim_conv=True):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_dimension = time_dim
        self.equal_dim_conv = equal_dim_conv
        self.encoder_channels_start = 64
        self.encoder_channels_end = self.encoder_channels_start * 4

        self.encoder = Encoder(input_channels, self.encoder_channels_start)
        self.bottleneck = Bottleneck(self.encoder_channels_end, equal_dim_conv)
        self.decoder = Decoder(self.encoder_channels_end, output_channels)

    def time_embeddings(self, t, time_channels):
        """
        time_embeddings encodes the timestamp information t as a tensor of dimension [B, time_channels]
        :param t: timestamp we are encoding, must in range [0, T]
        :param time_channels: dimension of generated time embeddings
        :return: time embedding's tensor, encodes the timestamp scalars as a [B, time_channels] tensor
        """
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, time_channels, 2, device=one_param(self).device).float() / time_channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, time_channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, time_channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forward(self, x, t):
        # Shape of x [B, 3, I=32, I=32], t [B, time_dimension=256]
        x1, x2, x3, x4 = self.encoder.forward(x, t)
        # Shape of x1 [B, 64, 32, 32], x2 [B, 128, 16, 16], x3 [1B, 256, 8, 8], x4 [B, 256, 4, 4]
        x4 = self.bottleneck(x4)
        # Shape of x4 [B, 256, 4, 4]

        out = self.decoder(x1, x2, x3, x4, t)
        # Shape of out [B, 3, 32, 32]
        return out

    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.time_embeddings(t, self.time_dimension)
        return self.unet_forward(x, t)


class UNetModuleConditional(UNetModule):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1)
        t = self.time_embeddings(t, self.time_dimension)

        if y is not None:
            t += self.label_emb(y)

        return self.unet_forward(x, t)
