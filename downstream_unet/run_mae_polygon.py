import os
import sys
import subprocess
import argparse
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 解决 PyTorch 2.6+ 默认 weights_only=True 导致加载 numpy 数据集报错的问题
# 将 numpy.ndarray 及内部重构函数加入安全全局变量白名单，适配所有 DDP 子进程
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        # 允许基础的 numpy 数组对象被加载
        torch.serialization.add_safe_globals([np.ndarray])
        
        # 兼容不同 numpy 版本的内部重建函数和标量
        if hasattr(np, '_core'):
            import numpy._core.multiarray
            torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
            if hasattr(numpy._core.multiarray, 'scalar'):
                torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
        elif hasattr(np, 'core'):
            import numpy.core.multiarray
            torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
            if hasattr(numpy.core.multiarray, 'scalar'):
                torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
                
        # 将基础 numpy.dtype 加入白名单
        torch.serialization.add_safe_globals([np.dtype])
        
        # 【新增修复】显式将具体的 numpy DType 子类（如 Float32DType）加入白名单
        if hasattr(np, 'dtypes'):
            for dtype_class in ['Float32DType', 'Float64DType', 'Int32DType', 'Int64DType']:
                if hasattr(np.dtypes, dtype_class):
                    torch.serialization.add_safe_globals([getattr(np.dtypes, dtype_class)])
        
        # 万能兜底策略：直接通过实例化反推真实类型并加入白名单
        torch.serialization.add_safe_globals([
            type(np.dtype(np.float32)), 
            type(np.dtype(np.float64)),
            np.float32, 
            np.float64
        ])
    except Exception:
        pass
# ---------------------------------------------------------

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from loaders.dataloader_mae import PolyMAEDataset, mae_collate_fn
from mae_pretrain.src.datasets.geometry_polygon import PolyFourierConverter
from models.vit_poly import MaskedAutoencoderViTPoly
from tqdm import tqdm

def plot_reconstruction(imgs, pred, mask, patch_size, epoch, save_dir):
    """可视化重建结果: 原始、Mask、重建，统一Colorbar，空掩码置NaN"""
    p = patch_size
    h, w = imgs.shape[2], imgs.shape[3]
    h_p, w_p = h // p, w // p
    
    # 获取单张图片
    img = imgs[0].detach().cpu()
    pred_img = pred[0].detach().cpu().reshape(h_p, w_p, 2, p, p)
    pred_img = torch.einsum('hwcpq->chpwq', pred_img).reshape(2, h, w)
    
    mask_map = mask[0].detach().cpu().reshape(h_p, w_p).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, p, p)
    mask_map = mask_map.reshape(h, w)
    
    # 构造带掩码的输入
    img_masked = img.clone()
    img_masked[:, mask_map == 1] = torch.nan
    
    # 构造完整重建图
    img_recon = img.clone()
    img_recon[:, mask_map == 1] = pred_img[:, mask_map == 1]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    titles = ['Original Mag', 'Masked Mag', 'Reconstructed Mag', 
              'Original Phase', 'Masked Phase', 'Reconstructed Phase']
    
    vmin_mag, vmax_mag = img[0].min(), img[0].max()
    vmin_ph, vmax_ph = img[1].min(), img[1].max()
    
    plot_data = [
        (img[0], vmin_mag, vmax_mag), (img_masked[0], vmin_mag, vmax_mag), (img_recon[0], vmin_mag, vmax_mag),
        (img[1], vmin_ph, vmax_ph), (img_masked[1], vmin_ph, vmax_ph), (img_recon[1], vmin_ph, vmax_ph)
    ]
    
    for i, ax in enumerate(axes.flatten()):
        data, vmin, vmax = plot_data[i]
        im = ax.imshow(data.numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(titles[i])
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'recon_epoch_{epoch}.png'))
    plt.close()

def main(args):
    # 智能检查是否为 DDP 模式 (依据 torchrun 注入环境变量)
    is_ddp = "LOCAL_RANK" in os.environ
    
    if is_ddp:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {local_rank}] 启动 DDP 多卡模式，使用设备: {device}")
    else:
        local_rank = 0
        if torch.cuda.is_available():
            # 因为我们在外部已经设置了 CUDA_VISIBLE_DEVICES，物理显卡已被映射为逻辑 0 号
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
            print(f"启动单卡模式，使用物理设备: GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
        else:
            device = torch.device("cpu")
            print("启动单卡模式，使用设备: CPU")

    # 1. 引擎与模型初始化
    fourier_engine = PolyFourierConverter(
        pos_freqs=args.pos_freqs, w_min=args.w_min, w_max=args.w_max, 
        freq_type=args.freq_type, device=device, patch_size=args.patch_size
    )
    
    img_size = (fourier_engine.U.shape[0], fourier_engine.U.shape[1])
    
    model = MaskedAutoencoderViTPoly(
        img_size=img_size, patch_size=args.patch_size,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        decoder_embed_dim=args.dec_embed_dim, decoder_depth=args.dec_depth, decoder_num_heads=args.dec_num_heads
    ).to(device)
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 2. 数据集初始化 (在CPU端)
    dataset = PolyMAEDataset(args.data_path, augment_times=args.augment_times)
    
    if is_ddp:
        sampler = DistributedSampler(dataset)
        # 将 num_workers 降低为 4 减轻多进程开销
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, 
                                collate_fn=mae_collate_fn, num_workers=4, pin_memory=True)
    else:
        sampler = None
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=mae_collate_fn, num_workers=4, pin_memory=True)

    # 3. 训练循环
    for epoch in range(args.epochs):
        if is_ddp:
            sampler.set_epoch(epoch)
        model.train()
        
        total_loss, total_mag_loss, total_phase_loss = 0, 0, 0
        start_time = time.time()
        
        # 只在主进程显示 tqdm 进度条，避免终端输出混乱
        if local_rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = dataloader
            
        for batch_tris, lengths in pbar:
            optimizer.zero_grad()
            
            # 第一步：实时GPU高并发傅里叶变换 (添加 no_grad() 避免构建庞大的无用计算图，节省海量显存和时间)
            with torch.no_grad():
                mag, phase = fourier_engine.cft_polygon_batch(batch_tris, lengths)
            
            # 第二步：MAE 前向传播
            _, loss_mag, loss_phase, pred, mask = model(mag, phase, mask_ratio=args.mask_ratio)
            
            # 使用自定义权重组合最终 Loss
            loss = args.weight_mag * loss_mag + args.weight_phase * loss_phase
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mag_loss += loss_mag.item()
            total_phase_loss += loss_phase.item()
            
            # 实时更新进度条上的 Loss 状态
            if local_rank == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # 数据同步以便精准打印 (单卡模式下则跳过这部分同步)
        avg_loss = torch.tensor(total_loss / len(dataloader), device=device)
        avg_mag_loss = torch.tensor(total_mag_loss / len(dataloader), device=device)
        avg_ph_loss = torch.tensor(total_phase_loss / len(dataloader), device=device)
        
        if is_ddp:
            torch.distributed.all_reduce(avg_loss)
            torch.distributed.all_reduce(avg_mag_loss)
            torch.distributed.all_reduce(avg_ph_loss)
            world_size = torch.distributed.get_world_size()
            avg_loss /= world_size
            avg_mag_loss /= world_size
            avg_ph_loss /= world_size
        
        if local_rank == 0:
            print(f"Epoch: {epoch+1}/{args.epochs} | Time: {time.time()-start_time:.2f}s | "
                  f"Total Loss: {avg_loss.item():.4f} | "
                  f"Mag Loss: {avg_mag_loss.item():.4f} | "
                  f"Phase Loss: {avg_ph_loss.item():.4f}")
            
            # 可视化 (利用最后一次batch的数据)
            os.makedirs(args.save_dir, exist_ok=True)
            imgs = torch.cat([mag, phase], dim=1)
            plot_reconstruction(imgs, pred, mask, args.patch_size, epoch+1, args.save_dir)
            
            # 保存 checkpoint (DDP环境下访问真实的model参数需要.module)
            if (epoch + 1) % 10 == 0:
                model_to_save = model.module if is_ddp else model
                torch.save(model_to_save.state_dict(), os.path.join(args.save_dir, f'ckpt_{epoch+1}.pth'))

    # 安全地关闭 DDP 进程组避免泄漏
    if is_ddp:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Poly2Vec MAE Training")
    parser.add_argument('--data_path', default='./data/polygon_triangles_normalized.pt', type=str)
    parser.add_argument('--save_dir', default='./checkpoints/', type=str)
    
    # 增加GPU设定，支持脱离 torchrun 直接使用字符串指定单卡或多卡
    parser.add_argument('--gpu', default="0,1,2,3", type=str, help='指定使用的GPU序号，多卡用逗号分隔，如 "1,2,3,4"')
    
    # 增加Loss的加权系数参数
    parser.add_argument('--weight_mag', default=1.0, type=float, help='幅值Loss的权重')
    parser.add_argument('--weight_phase', default=1.0, type=float, help='相位Loss的权重')
    
    # CFT频域参数
    parser.add_argument('--pos_freqs', default=31, type=int)
    parser.add_argument('--w_min', default=0.1, type=float)
    parser.add_argument('--w_max', default=100.0, type=float)
    parser.add_argument('--freq_type', default='geometric', type=str)
    
    # ViT架构参数 (默认设为 ViT-Small)
    parser.add_argument('--patch_size', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--depth', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dec_embed_dim', default=128, type=int)
    parser.add_argument('--dec_depth', default=4, type=int)
    parser.add_argument('--dec_num_heads', default=4, type=int)
    
    # 训练与MA参数
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int) # 单卡 Batch Size
    parser.add_argument('--lr', default=1.5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--augment_times', default=10, type=int)
    
    args = parser.parse_args()
    
    # === 智能环境准备及 DDP 拉起 ===
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    gpus = str(args.gpu).split(',')
    if len(gpus) > 1 and "LOCAL_RANK" not in os.environ:
        print(f"检测到您指定了多个GPU ({args.gpu})，正在自动为您启动 torchrun DDP 模式...")
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(len(gpus)), sys.argv[0]]
        # 将原有的其他命令行参数原封不动传递给子进程
        cmd.extend(sys.argv[1:])
        subprocess.run(cmd)
        sys.exit(0)
        
    main(args)