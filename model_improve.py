from typing_extensions import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.fft as fft

def inv_mag(x):
    fft_ = torch.fft.fft2(x)
    fft_ = torch.fft.ifft2(1 * torch.exp(1j * (fft_.angle())))
    return fft_.real


class AGSSF(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(AGSSF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=self.kernel_size(),
            padding=(self.kernel_size() - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        return k if k % 2 else k + 1

    def forward(self, x):
        x1 = inv_mag(x)
        y = self.avg_pool(x1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SFCA(nn.Module):
    def __init__(self, channels):
        super(SFCA, self).__init__()
        self.identity1 = nn.Conv2d(channels, channels, 1)
        self.identity2 = nn.Conv2d(channels, channels, 1)
        self.conv_1 = nn.Conv2d(channels, 2 * channels, 1, bias=True)
        self.relu_1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(2 * channels, channels, 3, padding=1, groups=channels, bias=True)
        self.relu_2 = nn.PReLU()

        self.conv_f1 = nn.Conv2d(channels, channels, 1)
        self.conv_f2 = nn.Conv2d(channels, channels, 1)
        self.con2X1 = nn.Conv2d(2 * channels, channels, 1)
        self.agssf = AGSSF(channels)

    def forward(self, x):
        out = self.relu_2(self.conv_2(self.relu_1(self.conv_1(x))))
        out = self.agssf(out) + self.identity1(x)

        x_fft = fft.fftn(x, dim=(-2, -1)).real
        x_fft = F.gelu(self.conv_f1(x_fft))
        x_fft = self.conv_f2(x_fft)
        x_reconstructed = fft.ifftn(x_fft, dim=(-2, -1)).real
        x_reconstructed = self.agssf(x_reconstructed) + self.identity2(x)
        f_out = self.con2X1(torch.cat([out, x_reconstructed], dim=1))
        return f_out


class MDTA(nn.Module): # MQCA
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)

        self.kv = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.q1X1_1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.q1X1_2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.kv_conv = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2, bias=False)
        self.project_outf = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))

        x_fft = fft.fftn(x, dim=(-2, -1)).real
        x_fft = self.q1X1_2(F.gelu(self.q1X1_1(x_fft)))
        qf = fft.ifftn(x_fft, dim=(-2, -1)).real
        kf, vf = self.kv_conv(self.kv(out)).chunk(2, dim=1)
        qf = qf.reshape(b, self.num_heads, -1, h * w)
        kf = kf.reshape(b, self.num_heads, -1, h * w)
        vf = vf.reshape(b, self.num_heads, -1, h * w)
        qf, kf = F.normalize(qf, dim=-1), F.normalize(kf, dim=-1)
        attnf = torch.softmax(torch.matmul(qf, k.transpose(-2, -1)) * self.temperature, dim=-1)
        outf = self.project_outf(torch.matmul(attn, vf).reshape(b, -1, h, w))
        return outf


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        hidden = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden * 2, 1, bias=False)
        self.conv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1))
                          .transpose(-2, -1).reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1))
                         .transpose(-2, -1).reshape(b, c, h, w))
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels, channel_red):
        super(UpSample, self).__init__()
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1),
                                      nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1),
                                      nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1))
        self.post = nn.Conv2d(channels, channels // 2 if channel_red else channels, 1)

    def forward(self, x):
        fft_x = torch.fft.fft2(x)
        mag_x, pha_x = torch.abs(fft_x), torch.angle(fft_x)
        Mag, Pha = self.amp_fuse(mag_x), self.pha_fuse(pha_x)
        amp, pha = torch.tile(Mag, (2, 2)), torch.tile(Pha, (2, 2))
        out = torch.complex(amp * torch.cos(pha), amp * torch.sin(pha))
        return self.post(torch.abs(torch.fft.ifft2(out)))


class UpSample1(nn.Module):
    def __init__(self, channels):
        super(UpSample1, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class UpS(nn.Module):
    def __init__(self, channels):
        super(UpS, self).__init__()
        self.Fups = UpSample(channels, True)
        self.Sups = UpSample1(channels)
        self.reduce = nn.Conv2d(channels, channels // 2, 1, bias=False)

    def forward(self, x):
        return self.reduce(torch.cat([self.Fups(x), self.Sups(x)], dim=1))


# FeatureGate 

class Gate(nn.Module):
    def __init__(self, channels, thresh=0.6, softness=0.15, strength_train=0.08, strength_eval=0.18):
        super().__init__()
        self.proj = nn.Conv2d(channels, 1, 1, bias=True)
        self.norm = nn.GroupNorm(1, 1)
        self.thresh = thresh
        self.softness = softness
        self.strength_train = strength_train
        self.strength_eval = strength_eval

    def forward(self, x):
        m = self.norm(self.proj(x))
        gate = torch.sigmoid((m - self.thresh) / self.softness)
        if self.training:
            strength = self.strength_train
        else:
            strength = self.strength_eval

        return x - strength * gate * x



class mymodel(nn.Module):
    def __init__(self, num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8],
                 channels=[16, 32, 64, 128], num_refinement=4,
                 expansion_factor=2.66, ch=[64, 32, 16, 64]):
        super(mymodel, self).__init__()

        self.attention = nn.ModuleList([SFCA(num_ch) for num_ch in ch])
        self.embed_conv_rgb = nn.Conv2d(3, channels[0], 3, padding=1, bias=False)
        self.encoders = nn.ModuleList([
            nn.Sequential(*[TransformerBlock(num_ch, num_ah, expansion_factor)
                            for _ in range(num_tb)])
            for num_tb, num_ah, num_ch in zip(num_blocks, num_heads, channels)
        ])

        self.down1 = DownSample(channels[0])
        self.down2 = DownSample(channels[1])
        self.down3 = DownSample(channels[2])
        self.ups_1 = UpS(128)
        self.ups_2 = UpS(64)
        self.ups_3 = UpS(32)
        self.ups1 = UpSample1(32)

        self.reduces1 = nn.Conv2d(128, 64, 1, bias=False)
        self.reduces2 = nn.Conv2d(64, 32, 1, bias=False)

        self.decoders = nn.ModuleList([
            nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                            for _ in range(num_blocks[2])]),
            nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                            for _ in range(num_blocks[1])]),
            nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                            for _ in range(num_blocks[0])])
        ])
        '''
        self.refinement = nn.Sequential(*[
            TransformerBlock(channels[1], num_heads[0], expansion_factor)
            for _ in range(num_refinement)
        ])
        '''
        self.refinement = nn.Sequential(
                *[nn.Sequential(TransformerBlock(channels[1], num_heads[0], expansion_factor),
                    SFCA(channels[1])) for _ in range(num_refinement)]
        )

        self.outputl = nn.Conv2d(32, 8, 3, padding=1, bias=False)
        self.output = nn.Conv2d(8, 3, 3, padding=1, bias=False)

        # === 新增亮度門控 ===
        self.expo_gate = Gate(channels[1])

    def forward(self, RGB_input):
        fo_rgb = self.embed_conv_rgb(RGB_input)
        e1 = self.encoders[0](fo_rgb)
        e2 = self.encoders[1](self.down1(e1))
        e3 = self.encoders[2](self.down2(e2))
        e4 = self.encoders[3](self.down3(e3))

        d3 = self.decoders[0](self.reduces1(torch.cat([self.ups_1(e4), self.attention[0](e3)], dim=1)))
        d2 = self.decoders[1](self.reduces2(torch.cat([self.ups_2(d3), self.attention[1](e2)], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups_3(d2), self.attention[2](e1)], dim=1))

        fr = self.refinement(fd)
        fr = self.expo_gate(fr)   # 特徵域亮度壓制
        return self.output(self.outputl(fr))
