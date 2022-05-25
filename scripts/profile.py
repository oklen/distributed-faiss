# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import numpy as np
import faiss
import time
from tqdm import tqdm
from pathlib import Path
import pickle
import hydra

import unittest
import torch
import random
import string

from distributed_faiss.rpc import Client
from distributed_faiss.server import IndexServer, DEFAULT_PORT
from distributed_faiss.client import IndexClient
from distributed_faiss.index_state import IndexState
from distributed_faiss.index_cfg import IndexCfg
import time

import json
import logging

logging.basicConfig(level=4)




def array_to_memmap(array, filename):
    if os.path.exists(filename):
        fp = np.memmap(filename, mode="r", dtype=array.dtype, shape=array.shape)
        return fp

    fp = np.memmap(filename, mode="write", dtype=array.dtype, shape=array.shape)
    fp[:] = array[:]  # copy
    fp.flush()
    del array
    return fp


def save_random_mmap(path, nrow, ncol, chunk_size=100000):
    fp = np.memmap(path, mode="write", dtype=np.float16, shape=(nrow, ncol))

    for i in tqdm(range(0, nrow, chunk_size), desc=f"saving random mmap to {path}"):
        end = min(nrow, i + chunk_size)
        fp[i:end] = np.random.rand(end - i, ncol).astype(fp.dtype)
    fp.flush()
    del fp


"""
python scripts/load_data.py --discover discover_val.txt \
    --mmap random --mmap-size 111649041 --dimension 768 \
    --cfg idx_cfg.json
"""
def get_quantizer(cfg: IndexCfg):
    metric = cfg.get_metric()
    if metric == faiss.METRIC_INNER_PRODUCT:
        quantizer = faiss.IndexFlatIP(cfg.dim)
    elif metric == faiss.METRIC_L2:
        quantizer = faiss.IndexFlatL2(cfg.dim)
    else:
        raise RuntimeError(f"Metric={metric} is not supported")
    return quantizer

@hydra.main(config_path="conf", config_name="server")
def main(args):


    cfg = IndexCfg(
        index_builder_type=args.index_builder_type,
        dim=args.dim,
        train_num=args.train_num,
        centroids=args.centroids,
        metric=args.metric,
        # nprobe=args.nprobe,
        nprobe=args.centroids,
        index_storage_dir=args.index_storage_dir,
    )

    client = IndexClient(args.discovery_config, cfg=cfg)

    index_id = "wiki"

    # client.create_index(index_id, cfg)
    index_id_to_db_id = []

    total_time_cost = 0

    n = 2000
    # rand_vec = torch.rand((1, 768)).numpy()
    # d = client.search(rand_vec, 4, index_id, return_embeddings=True)
    # for i in range(n):
    #     rand_vec = torch.rand((1, 768)).numpy()
    #     time0 = time.time()
    #     res = client.search(rand_vec, 5, index_id, return_embeddings=True)
    #     # res = client.search(rand_vec, 4, index_id)
    #     total_time_cost += time.time() - time0
    # print("1 * 5 index search time: %f sec.", total_time_cost / n)

    # client.save_index(index_id)

    print(f"ntotal: {client.get_ntotal(index_id)}")

    for i in range(n):
        rand_vec = torch.rand((4, 768)).numpy()
        time0 = time.time()
        d = client.search(rand_vec, 5, index_id, return_embeddings=True)
        once_time = time.time() - time0
        print("4*5:" ,once_time)
        total_time_cost += once_time
    print("4 * 5 index search time: %f sec.", total_time_cost / n)

    for i in range(n):
        rand_vec = torch.rand((32, 768)).numpy()
        time0 = time.time()
        d = client.search(rand_vec, 5, index_id, return_embeddings=True)
        once_time = time.time() - time0
        print("32*5:", once_time)
        total_time_cost += once_time
    print("32 * 5 index search time: %f sec.", total_time_cost / n)
if __name__ == "__main__":
    main()
