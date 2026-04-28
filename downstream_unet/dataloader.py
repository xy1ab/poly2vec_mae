#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataloader.py
@Time    :   2026/04/25 15:24:38
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
import webdataset as wds
import glob
import hashlib
import torch.distributed as dist
from functools import partial
import json

def load_samples(index_file):
    with open(index_file, "r") as f:
        index = json.load(f)
        train_info = index['train']
        val_info = index['val']        
    return train_info, val_info


def sample_belongs_to_rank(sample, rank, world_size):
    key = sample["__key__"]
    if not isinstance(key, str):
        key = key.decode("utf-8")
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    sample_rank = int.from_bytes(digest, "little") % world_size
    return sample_rank == rank

def get_wds_loader(url_pattern, batch_size, total_samples, is_training=True, num_workers=4, split_by_rank=True, split_samples_by_rank=False):
    file_list = sorted(glob.glob(url_pattern))
    
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()} found {len(file_list)} shards for {url_pattern}")
        
    if not file_list:
        raise FileNotFoundError(f"未找到匹配的文件: {url_pattern}")

    # nodesplitter=wds.split_by_node 自动实现 DDP 分片
    shard_shuffle_buffer = 5000 if is_training else 0
    nodesplitter = wds.split_by_node if split_by_rank else None
    dataset = wds.WebDataset(file_list, shardshuffle=shard_shuffle_buffer, nodesplitter=nodesplitter, empty_check=False)

    if dist.is_initialized() and split_samples_by_rank:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dataset = dataset.select(partial(sample_belongs_to_rank, rank=rank, world_size=world_size))

    dataset = (
        dataset
        .decode("torch")
        .to_tuple("input.npy", "label.npy") # 根据你保存的 key
        .batched(batch_size, partial=not is_training)
    )

    epoch_batches = None
    if dist.is_initialized() and is_training:
        world_size = dist.get_world_size() if split_by_rank else 1
        epoch_batches = total_samples // (batch_size * world_size)
        if epoch_batches < 1:
            raise ValueError("训练样本数小于全局 batch size，无法在 drop_last=True 下安全启动 DDP")

    # WDS 已经完成了 batch 组装，DataLoader 的 batch_size 需设为 None
    loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=True)
    if epoch_batches is not None:
        loader = loader.with_epoch(nbatches=epoch_batches)
    return loader


if __name__ == '__main__':
    pass

