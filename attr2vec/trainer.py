import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from data_loader import load_and_preprocess_gdb
from models import NaturalResourceFoundationModel

torch.set_default_dtype(torch.float32)

class GDBDataset(Dataset):
    # 🌟 修复：接入最新架构的 4 路数据
    def __init__(self, norm, c_int, c_frac, cat):
        self.norm = norm
        self.c_int = c_int
        self.c_frac = c_frac
        self.cat = cat
    def __len__(self): return len(self.norm)
    def __getitem__(self, idx): return self.norm[idx], self.c_int[idx], self.c_frac[idx], self.cat[idx]

def train():
    is_distributed = "LOCAL_RANK" in os.environ

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1

    if local_rank == 0:
        mode_str = f"DDP {world_size} 卡并行" if is_distributed else "单卡直驱"
        print("="*115)
        print(f"🚀 启动自然资源大模型 [0.1B规模 | {mode_str} | 单向cINN护甲] 预训练")
        print("="*115)
    
    # 获取数据
    data = load_and_preprocess_gdb("/home/xz/myq/NRE2/data/LCXZ_TEST.gdb", "LCXZ_TEST01_XZ")
    
    # 🌟 修复：提取解耦后的精确张量
    dataset = GDBDataset(
        torch.tensor(data['cont_norm']), 
        torch.tensor(data['cont_int']), 
        torch.tensor(data['cont_frac']), 
        torch.tensor(data['cat_data'])
    )
    
    BATCH_SIZE = 512
    sampler = DistributedSampler(dataset) if is_distributed else None
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=(sampler is None))
    
    model = NaturalResourceFoundationModel(
        num_cont_cols=len(data['cont_names']), 
        cat_cardinalities=data['cat_cardinalities']
    ).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    cont_criterion = nn.MSELoss()
    cat_criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    
    for epoch in range(1000):
        if is_distributed: sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if local_rank == 0 else dataloader
        
        for bn, bi, bf, bc in pbar:
            # 🌟 四路张量入显存
            bn, bi, bf, bc = bn.to(device), bi.to(device), bf.to(device), bc.to(device)
            optimizer.zero_grad()
            
            # 压入模型
            _, _, p_cont, p_cat, _ = model(bn, bi, bf, bc, mask_ratio=0.25)
            
            loss = 0
            if p_cont is not None:
                # 语义特征依然向 Standard 归一化后的特征对齐，保持梯度平稳
                loss += cont_criterion(p_cont, bn)
            
            for i, pred in enumerate(p_cat):
                vocab_size = pred.shape[1]
                target = bc[:, i].long() % vocab_size
                loss += cat_criterion(pred, target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if local_rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        
        if is_distributed:
            avg_loss = total_loss / len(dataloader)
            loss_tensor = torch.tensor([avg_loss]).to(device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss_final = loss_tensor.item() / world_size
        else:
            avg_loss_final = total_loss / len(dataloader)

        if local_rank == 0:
            print(f"👉 轮次总结: 全局平均 Loss: {avg_loss_final:.4f} | 当前 LR: {scheduler.get_last_lr()[0]:.2e}")
            if avg_loss_final < best_loss:
                best_loss = avg_loss_final
                state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save(state_dict, "natural_resource_0.1B_512dim.pth")
                print(f"   🎯 发现更优 Loss ({best_loss:.4f})，权重已保存。")

    if is_distributed: dist.destroy_process_group()

if __name__ == "__main__":
    train()