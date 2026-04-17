import os
import glob
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.amp import autocast, GradScaler

from config import ModelConfig
from models import Attr2Vec

# ======================================================================
# 1. 数据集加载器 (直接读取缓存的 PT 文件)
# ======================================================================
class NRECacheDataset(Dataset):
    def __init__(self, cache_dir):
        super().__init__()
        self.truth_vectors = []
        self.string_ids = []
        
        pt_files = glob.glob(os.path.join(cache_dir, "cache_*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"❌ 在 {cache_dir} 未找到任何缓存文件，请先运行 data_loader.py")
            
        print(f"📦 正在汇聚 {len(pt_files)} 个缓存文件中的张量数据...")
        
        for file in pt_files:
            try:
                data_dict = torch.load(file, weights_only=False)
                for layer_name, tensors in data_dict.items():
                    tv = tensors.get('truth_vector')
                    si = tensors.get('string_ids')
                    
                    if tv is not None and si is not None:
                        self.truth_vectors.append(torch.tensor(tv, dtype=torch.float32))
                        self.string_ids.append(torch.tensor(si, dtype=torch.long))
            except Exception as e:
                print(f"⚠️ 读取 {file} 失败，已跳过: {e}")

        self.truth_vectors = torch.cat(self.truth_vectors, dim=0)
        self.string_ids = torch.cat(self.string_ids, dim=0)
        self.total_samples = self.truth_vectors.shape[0]
        print(f"✅ 数据集装载完毕！总样本数: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return self.truth_vectors[idx], self.string_ids[idx]

# ======================================================================
# 2. 分布式环境初始化
# ======================================================================
def setup_ddp():  
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    return local_rank, global_rank

def cleanup_ddp():
    dist.destroy_process_group()

# ======================================================================
# 3. 核心训练主循环
# ======================================================================
def train():

    
    # 🌟 1. 启动配置解析 (新增 batch_size 外部调节开关)
    parser = argparse.ArgumentParser(description="ZrZy Foundation Model Training")
    parser.add_argument("--config_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/attr/config.json", help="存放 cache_*.pt 的目录")
    parser.add_argument("--cache_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/attr/", help="存放 cache_*.pt 的目录")
    parser.add_argument("--output_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/attr", help="模型权重和日志保存目录")
    parser.add_argument("--batch_size", type=int, default=256, help="单卡 Batch Size (降低此值以解决 OOM)")
    args = parser.parse_args()
    
    config = ModelConfig()
    config.load(args.config_path)
        
    # 🌟 动态覆盖 Batch Size 以防止显存溢出
    config.batch_size = args.batch_size
    local_rank, global_rank = setup_ddp()
    if global_rank == 0:
        print("\n" + "="*60)
        print("🚀 [南湖大一统底座] 物理/语义解耦架构开始训练")
        print(f"👉 物理真值轨道维度: {config.truth_dim} (已冻结梯度)")
        print(f"👉 语义提取轨道维度: {config.semantic_dim} (独立训练 MAE)")
        print(f"👉 硬件对齐输出维度: {config.final_dim} 维")
        print(f"👉 当前单卡 Batch Size: {config.batch_size}")
        print("="*60 + "\n")

    # 2. 构建模型与分布式包裹
    model = Attr2Vec(config).cuda(local_rank)
    # 🌟 开启未使用参数检测，允许 to_semantic 头在预训练时暂不更新
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=config.base_lr, weight_decay=0.05)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler('cuda')

    # 3. 加载数据集
    dataset = NRECacheDataset(args.cache_dir)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True, 
    )

    # 4. 开始训练
    best_loss = float('inf')
    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        
        start_time = time.time()
        for batch_idx, (truth_vec, str_ids) in enumerate(dataloader):
            truth_vec = truth_vec.cuda(local_rank, non_blocking=True)
            str_ids = str_ids.cuda(local_rank, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                _, mae_loss = model(truth_vec, str_ids)
                loss = mae_loss.mean()
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        
        # 5. 日志与模型保存
        if global_rank == 0:
            avg_loss = epoch_loss / len(dataloader)
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{config.epochs}] | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.2f}s")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt_path = os.path.join(args.output_dir, "zrzy_foundation_best.pth")
                torch.save(model.module.state_dict(), ckpt_path)
                print(f"   💾 发现更低 Loss，已保存权重至 {ckpt_path}")

    cleanup_ddp()

if __name__ == "__main__":
    train()