import torch
import torch.nn as nn
import numpy as np
from shapely.geometry import Polygon
import triangle as tr


def build_poly_fourier_converter_from_config(config, device='cpu'):
    return PolyFourierConverter(
        pos_freqs=config.get("pos_freqs", 31),
        w_min=config.get("w_min", 0.1),
        w_max=config.get("w_max", 100.0),
        freq_type=config.get("freq_type", 'geometric'),
        patch_size=config.get("patch_size", 2),
        device=device,
    )

class PolyFourierConverter(nn.Module):
    """
    提供面向Polygon的连续傅里叶变换 (CFT) 和反变换 (ICFT) 的核心类。
    支持在GPU上对批次多边形进行实时高并发的CFT计算。
    """
    def __init__(self, pos_freqs=31, w_min=0.1, w_max=1.0, freq_type='geometric', device='cuda', patch_size=16):
        super().__init__()
        self.pos_freqs = pos_freqs
        self.w_min = w_min
        self.w_max = w_max
        self.freq_type = freq_type
        self.device = torch.device(device)
        self.patch_size = patch_size
        
        self.U, self.V, self.pad_h, self.pad_w = self._build_meshgrid()

    def _build_meshgrid(self):
        """构建1/2象限的频率网格，并计算自动补全参数"""
        if self.freq_type == 'geometric':
            # 几何级数
            g = (self.w_max / self.w_min) ** (1 / (self.pos_freqs - 1)) if self.pos_freqs > 1 else 1.0
            pos_w = [self.w_min * (g ** u) for u in range(self.pos_freqs)]
        else:
            # 等差级数
            pos_w = np.linspace(self.w_min, self.w_max, self.pos_freqs).tolist()
            
        pos_w = torch.tensor(pos_w, dtype=torch.float32)
        
        # Wx 维度 (u): 包含负频率, 0, 正频率 -> 长度 2 * pos_freqs + 1
        Wx = torch.cat((-torch.flip(pos_w, dims=[0]), torch.tensor([0.0]), pos_w))
        # Wy 维度 (v): 根据对称性，只取1/2象限，即 0 和正频率 -> 长度 pos_freqs + 1
        Wy = torch.cat((torch.tensor([0.0]), pos_w))
        
        # 计算补全量，使其能被 patch_size 整除
        len_x, len_y = len(Wx), len(Wy)
        pad_h = (self.patch_size - (len_x % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (len_y % self.patch_size)) % self.patch_size
        
        U, V = torch.meshgrid(Wx, Wy, indexing='ij')
        
        # 扩展 U, V 进行补全 (频率置 0 避免干扰，实际会在补全区域填充 0)
        if pad_h > 0 or pad_w > 0:
            U = torch.nn.functional.pad(U, (0, pad_w, 0, pad_h), value=0.0)
            V = torch.nn.functional.pad(V, (0, pad_w, 0, pad_h), value=0.0)
            
        return U.to(self.device), V.to(self.device), pad_h, pad_w

    def triangulate_polygon(self, coords):
        """对单个 Polygon 的坐标系进行三角剖分"""
        poly = Polygon(coords).buffer(0) # 修复自相交
        if poly.is_empty or poly.geom_type != 'Polygon':
            return np.zeros((0, 3, 2), dtype=np.float32)
            
        exterior = list(poly.exterior.coords)[:-1]
        if len(exterior) < 3:
            return np.zeros((0, 3, 2), dtype=np.float32)
            
        segments = [(i, (i + 1) % len(exterior)) for i in range(len(exterior))]
        poly_dict = {'vertices': exterior, 'segments': segments}
        try:
            tri_data = tr.triangulate(poly_dict, 'pq') # constrained Delaunay
            triangles = tri_data['vertices'][tri_data['triangles']]
            return triangles.astype(np.float32)
        except Exception:
            return np.zeros((0, 3, 2), dtype=np.float32)

    def _cft_single_triangle_batch(self, tris, valid_mask):
        """计算批次中单个有效三角形的 CFT。输入tris: [B, 3, 2]"""
        pi = torch.pi
        U, V = self.U.unsqueeze(0), self.V.unsqueeze(0) # [1, H, W]
        
        # 仿射变换参数提取
        q, r, s = tris[:, 0, :], tris[:, 1, :], tris[:, 2, :]
        xq, yq = q[:, 0].view(-1, 1, 1), q[:, 1].view(-1, 1, 1)
        xr, yr = r[:, 0].view(-1, 1, 1), r[:, 1].view(-1, 1, 1)
        xs, ys = s[:, 0].view(-1, 1, 1), s[:, 1].view(-1, 1, 1)
        
        # 计算行列式 (面积的2倍)
        det = xq*(yr - ys) + xr*(ys - yq) + xs*(yq - yr)
        area = torch.abs(0.5 * det)
        
        # 修正: A^T [U, V]^T 频率直接映射，避免除以极小的det引发几十上百倍的错误频率畸变放大
        U_ = U * (xr - xq) + V * (yr - yq)
        V_ = U * (xs - xr) + V * (ys - yr)
        
        # 修正: 相移只需采用平移原点 q 即可
        phase_shift = torch.exp(-2j * pi * (U * xq + V * yq))
        
        # 处理解析极限 (0分母情况，严格排他避免计算崩溃)
        zero_mask = (U_ == 0) & (V_ == 0)
        mask = (U_ + V_ == 0) & ~zero_mask
        u_mask = (U_ == 0) & ~zero_mask
        v_mask = (V_ == 0) & ~zero_mask
        normal_mask = ~(mask | zero_mask | u_mask | v_mask)
        
        base_u = torch.exp(-2j * pi * U_)
        base_v = torch.exp(-2j * pi * V_)
        base_uv = torch.exp(-2j * pi * (U_ + V_))
        
        part1 = torch.zeros_like(U_, dtype=torch.complex64)
        part2 = torch.zeros_like(U_, dtype=torch.complex64)
        
        # Normal
        part1[normal_mask] = (1.0 / (4 * pi**2 * (U_[normal_mask]*V_[normal_mask]*(U_[normal_mask]+V_[normal_mask])))).to(torch.complex64)
        part2[normal_mask] = U_[normal_mask]*(-base_uv[normal_mask]) + (U_[normal_mask]+V_[normal_mask])*base_u[normal_mask] - V_[normal_mask]
        
        # U+V == 0
        part1[mask] = (-1.0 / (4 * pi**2 * U_[mask]**2)).to(torch.complex64)
        part2[mask] = base_u[mask] + 2j * pi * U_[mask] - 1.0
        
        # U == 0
        part1[u_mask] = (-1.0 / (4 * pi**2 * V_[u_mask]**2)).to(torch.complex64)
        part2[u_mask] = base_v[u_mask] + 2j * pi * V_[u_mask] - 1.0
        
        # V == 0
        part1[v_mask] = (1.0 / (4 * pi**2 * U_[v_mask]**2)).to(torch.complex64)
        part2[v_mask] = (2j * pi * U_[v_mask] + 1.0)*base_u[v_mask] - 1.0
        
        # 依据傅里叶变换的仿射特性，由于积分域变化，结果需要乘以 |det(A)| = 2 * area
        FT = 2.0 * area * part1 * part2 * phase_shift
        FT = torch.where(zero_mask, area.to(torch.complex64), FT)
        
        # 补全区域的频谱置为0
        if self.pad_h > 0:
            FT[:, -self.pad_h:, :] = 0.0
        if self.pad_w > 0:
            FT[:, :, -self.pad_w:] = 0.0
            
        return FT

    def cft_polygon_batch(self, batch_triangles, lengths):
        """
        高度优化的批处理 CFT 计算，使用块处理和 Scatter-add 彻底消除循环瓶颈
        """
        B, max_N, _, _ = batch_triangles.shape
        H, W = self.U.shape
        
        FT_total = torch.zeros((B, H, W), dtype=torch.complex64, device=self.device)
        batch_triangles = batch_triangles.to(self.device)
        lengths = lengths.to(self.device)
        
        # 1. 创建 [B, max_N] 的有效性掩码
        mask = torch.arange(max_N, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # 2. 扁平化提取所有有效的三角形，直接消灭 max_N Python 循环！
        valid_tris = batch_triangles[mask] # 形状: [Total_valid_tris, 3, 2]
        
        # 对应记录每个有效三角形属于原始批次中的哪一个索引 [Total_valid_tris]
        batch_indices = torch.arange(B, device=self.device).unsqueeze(1).expand(B, max_N)[mask]
        
        # 3. 分块并行处理 (Chunking)，防止一次性并发所有三角形导致 OOM
        chunk_size = 50000 
        for i in range(0, valid_tris.shape[0], chunk_size):
            tris_chunk = valid_tris[i:i+chunk_size]
            b_idx_chunk = batch_indices[i:i+chunk_size]
            
            # 直接计算这一大块三角形的 CFT (此处的 None 表示不需要 valid_mask)
            ft_chunk = self._cft_single_triangle_batch(tris_chunk, None) # 形状: [chunk, H, W]
            
            # 采用 PyTorch 极速的底层索引累加，加回到它们各自的 Batch 位置
            FT_total.index_add_(0, b_idx_chunk, ft_chunk)
            
        mag = torch.abs(FT_total).unsqueeze(1)    # [B, 1, H, W]
        phase = torch.angle(FT_total).unsqueeze(1) # [B, 1, H, W]
        
        # 归一化/稳定化幅值
        mag = torch.log1p(mag)
        
        return mag, phase

    def icft_2d(self, F_uv, spatial_size=256):
        """
        离散非均匀傅里叶逆变换 (ICFT)
        输入:
            F_uv: [B, H, W] complex tensor (原始未经过log1p的复数频谱)
            spatial_size: int, 生成的空域图像的分辨率
        输出:
            f_norm: [B, spatial_size, spatial_size] real tensor, 归一化到[0,1]用于可视化的空域图
        """
        B, H, W = F_uv.shape
        x = torch.linspace(-1, 1, spatial_size, device=self.device)
        y = torch.linspace(1, -1, spatial_size, device=self.device) # y轴向下匹配图像坐标系
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        X_flat = X.reshape(-1) # [S^2]
        Y_flat = Y.reshape(-1) # [S^2]
        U_flat = self.U.reshape(-1) # [H*W]
        V_flat = self.V.reshape(-1) # [H*W]
        
        # 1. 精确计算非均匀积分权重 (避免几何频域低频被过度放大)
        valid_h = H - self.pad_h
        valid_w = W - self.pad_w
        Wx = self.U[:valid_h, 0].clone()
        Wy = self.V[0, :valid_w].clone()
        
        du = torch.zeros_like(Wx)
        for i in range(valid_h):
            if i == 0: du[i] = Wx[1] - Wx[0] if valid_h > 1 else 1.0
            elif i == valid_h - 1: du[i] = Wx[i] - Wx[i-1]
            else: du[i] = (Wx[i+1] - Wx[i-1]) / 2.0
            
        dv = torch.zeros_like(Wy)
        for j in range(valid_w):
            if j == 0: dv[j] = Wy[1] - Wy[0] if valid_w > 1 else 1.0
            elif j == valid_w - 1: dv[j] = Wy[j] - Wy[j-1]
            else: dv[j] = (Wy[j+1] - Wy[j-1]) / 2.0
            
        dU, dV = torch.meshgrid(du, dv, indexing='ij')
        
        # 构建全平面积分权重，并去除pad部分
        weights = torch.zeros((H, W), device=self.device)
        weights[:valid_h, :valid_w] = dU * dV
        weights = weights.reshape(-1).unsqueeze(0) # [1, H*W]
        
        # 2. 处理1/2象限共轭对称
        # 当 V > 0 时，全平面还应该加上共轭部分 F(-u,-v) = F*(u,v)
        # 积分 f(x,y) = \sum_{v>0} (F*E + F**E*) + \sum_{v=0} F*E 
        #           = 2 * Re(\sum_{v>0} F*E) + \sum_{v=0} F*E
        # 为统合计算，我们将 v>0 部分的权重直接 x2，最后统一取实部即可
        sym_weights = torch.where(V_flat > 1e-6, 2.0, 1.0).unsqueeze(0) # [1, H*W]
        weights = weights * sym_weights
        
        # 3. 计算相位偏移矩阵 E
        # 相位 = 2 * pi * (U*X + V*Y)
        phase_mat = 2 * torch.pi * (U_flat.unsqueeze(1) * X_flat.unsqueeze(0) + V_flat.unsqueeze(1) * Y_flat.unsqueeze(0))
        E = torch.exp(1j * phase_mat) # [H*W, S^2]
        
        # 4. 矩阵乘法求解离散逆变换
        F_flat = F_uv.reshape(B, -1) * weights # [B, H*W]
        f_recon_complex = torch.matmul(F_flat, E) # [B, S^2]
        
        f_recon = f_recon_complex.real.reshape(B, spatial_size, spatial_size)
        
        # 5. 归一化到 [0, 1] 以便于二值化或直观对比
        f_min = f_recon.amin(dim=(1, 2), keepdim=True)
        f_max = f_recon.amax(dim=(1, 2), keepdim=True)
        f_norm = (f_recon - f_min) / (f_max - f_min + 1e-8)
        
        return f_norm
