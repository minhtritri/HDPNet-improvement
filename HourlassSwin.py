# HourglassSwin_Base.py
# --------------------------------------------
# Swin Transformer backbone for HDPNet (replaces PVTv2)
# Author: adaptation by ChatGPT (2025)
# --------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


class ConvProj(nn.Module):
    """Simple Conv projection to adjust channel dimensions"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.proj(x)


class HourglassSwin(nn.Module):
    """
    Hourglass-style Swin Transformer backbone for HDPNet.
    Generates 12 multi-scale token tensors matching HDPNet's expected input.

    Output:
        List[Tensor] with shapes [(B, N_i, C_i), ...], where sizes correspond to:
        [96,96,48,48,24,24,24,24,48,48,96,96]
    """
    def __init__(self, img_size=384,
                 model_name='swin_base_patch4_window7_224',
                 pretrained=True):
        super().__init__()

        # 1️⃣ Swin backbone from timm
        self.swin = create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,    # output list of feature maps
            out_indices=(0, 1, 2, 3)
        )

        # 2️⃣ Extract Swin feature channels (varies by model size)
        swin_chs = [f['num_chs'] for f in self.swin.feature_info]  # e.g. [128,256,512,1024]

        # 3️⃣ Project to HDPNet-compatible channels
        # (matches the paper: 64, 128, 320)
        self.proj1 = ConvProj(swin_chs[0], 64)    # for 96x96
        self.proj2 = ConvProj(swin_chs[1], 128)   # for 48x48
        self.proj3 = ConvProj(swin_chs[2], 320)   # for 24x24
        self.proj4 = ConvProj(swin_chs[3], 320)   # for 12x12 (upsampled)

        # 4️⃣ Refinement convs for variant maps
        self.refine_24a = nn.Conv2d(320, 320, 3, padding=1, bias=False)
        self.refine_24b = nn.Conv2d(320, 320, 3, padding=1, bias=False)
        self.refine_48a = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.refine_96a = nn.Conv2d(64, 64, 3, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 5️⃣ Extract 4 feature maps from Swin backbone
        f1, f2, f3, f4 = self.swin(x)
        # Example sizes for input 384x384:
        # f1: (B, C1, 96, 96)
        # f2: (B, C2, 48, 48)
        # f3: (B, C3, 24, 24)
        # f4: (B, C4, 12, 12)

        # 6️⃣ Project to unified channels
        f1 = self.proj1(f1)  # -> (B,64,96,96)
        f2 = self.proj2(f2)  # -> (B,128,48,48)
        f3 = self.proj3(f3)  # -> (B,320,24,24)
        f4 = self.proj4(f4)  # -> (B,320,12,12)

        # 7️⃣ Upsample deeper features to same scale
        f4_up = F.interpolate(f4, size=(24,24), mode='bilinear', align_corners=False)
        p3_up_48 = F.interpolate(f3, size=(48,48), mode='bilinear', align_corners=False)
        p2_up_96 = F.interpolate(f2, size=(96,96), mode='bilinear', align_corners=False)

        # 8️⃣ Generate multi-scale variants (Hourglass symmetry)
        map24_1 = self.act(self.refine_24a(f3))
        map24_2 = self.act(self.refine_24b(f4_up))
        map24_3 = self.act(self.refine_24a((f3 + f4_up) / 2))
        map24_4 = self.act(self.refine_24b(f3 + 0.5 * f4_up))

        map48_1 = self.act(self.refine_48a(f2))
        map48_2 = self.act(self.refine_48a(p3_up_48))

        map96_1 = self.act(self.refine_96a(f1))
        map96_2 = self.act(self.refine_96a(p2_up_96))

        # 9️⃣ Assemble the list of maps in order expected by HDPNet.cim_decoder
        maps = [
            map96_1, map96_2, map48_1, map48_2,
            map24_1, map24_2, map24_3, map24_4,
            map48_2, map48_1, map96_2, map96_1
        ]

        # 🔟 Convert feature maps to token-like tensors (B, N, C)
        outs = []
        for m in maps:
            b, c, h, w = m.size()
            outs.append(m.view(b, c, -1).permute(0, 2, 1).contiguous())
        return outs


def Hourglass_vision_transformer_base_swin():
    """Factory function (same pattern as Hourglass_vision_transformer_base_v2)"""
    model = HourglassSwin(img_size=384, model_name='swin_base_patch4_window7_224', pretrained=True)
    return model


# ✅ Quick test
if __name__ == "__main__":
    x = torch.randn(1, 3, 384, 384)
    net = Hourglass_vision_transformer_base_swin()
    outs = net(x)
    print(f"Output tokens: {len(outs)}")
    for i, o in enumerate(outs):
        print(f"{i}: {o.shape}")
