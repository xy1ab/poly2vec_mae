#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   decoder.py
@Time    :   2026/04/07 16:27:32
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class TransUNetdecoder(nn.Module):

    def __init__(self,embed_dim=1024, out_ch=1):
        super().__init__()

# 1. 维度对齐与全局信息融合
        # 这里的 512 是我们进入 Decoder 的基准维度
        self.spatial_proj = nn.Conv2d(embed_dim, 512, kernel_size=1)
        self.cls_proj = nn.Linear(embed_dim, 512)
        
        # 2. 定义 SMP Unet Decoder 结构
        # 级联上采样通道：512 -> 256 -> 128 -> 64 -> 32 -> 16
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=(0, 0, 0, 0, 512), 
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
        )

        # 3. 输出头 (Segmentation Head)
        # 最后输出 256x256，这里设置 upsampling=2 配合 decoder 内部的 4 倍
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=32, 
            out_channels=out_ch, 
            activation=None, # 训练建议：输出 Logits，配合 BCEWithLogitsLoss
            kernel_size=3
        )
    def forward(self, x):
            """
            x: [Batch, 513, 1024]
            """
            B, N, C = x.shape
            
            # --- 建议 1: 融合 [CLS] 全局信息 ---
            # cls_token: [B, 1, 1024] -> [B, 512, 1, 1]
            cls_feat = self.cls_proj(x[:, 0, :]).unsqueeze(-1).unsqueeze(-1)
            
            # spatial_patches: [B, 512, 1024] -> [B, 1024, 32, 16]
            x_spatial = x[:, 1:, :].transpose(1, 2).contiguous().reshape(B, C, 32, 16)
            x_spatial = self.spatial_proj(x_spatial) # [B, 512, 32, 16]
            
            # 全局语义与局部空间特征相加融合
            x_spatial = x_spatial + cls_feat 
            d1 = torch.zeros(B, 0, 512, 256).to(x.device)
            d2 = torch.zeros(B, 0, 256, 128).to(x.device)
            d3 = torch.zeros(B, 0, 128, 64).to(x.device)
            d4 = torch.zeros(B, 0, 64, 32).to(x.device)
            features = [d1, d2, d3, d4, x_spatial]
            # --- 建议 2: 纯 Decoder 解码 ---
            # 此时 features 是最底层的 Bottleneck 特征
            # 经过 5 层上采样：(32x16) -> (64x32) -> (128x64) -> (256x128) -> (512x256) -> (1024x512)
            # 注意：smp 默认每层放大2倍，所以我们需要控制上采样步数
            decoder_output = self.decoder(features)
            
            # --- 建议 3: 尺寸对齐与 Logits 输出 ---
            logits = self.segmentation_head(decoder_output)
            
            # 强制缩放到目标 256x256 (处理长宽比不一致问题)
            if logits.shape[-2:] != (256, 256):
                logits = torch.nn.functional.interpolate(
                    logits, size=(256, 256), mode='bilinear', align_corners=True
                )
                
            return logits # 输出 [B, 256, 256]
if __name__ == '__main__':
    pass

