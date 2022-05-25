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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmap", type=str, help="memmap where keys and vals are stored")
    parser.add_argument(
        "--mmap-size", type=int, help="number of items saved in the datastore memmap"
    )

    parser.add_argument("--dimension", type=int, default=1024, help="Size of each key")
    parser.add_argument("--cfg", type=str, default=None, help="path to index config json")
    parser.add_argument("--dstore-fp16", default=False, action="store_true")
    parser.add_argument(
        "--ncentroids",
        type=int,
        default=4096,
        help="number of centroids faiss should learn",
    )
    parser.add_argument(
        "--bs",
        default=1000,
        type=int,
        help="can only load a certain amount of data to memory at a time.",
    )
    parser.add_argument("--start", default=0, type=int, help="index to start adding keys at")
    parser.add_argument(
        "--discover",
        type=str,
        help="serverlist_path",
    )
    parser.add_argument("--load_index", action="store_true")
    args = parser.parse_args()
    return args


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

    # if args.mmap == "random":
    #     rand_path = f"random_{args.mmap_size}_{args.dimension}_fp16.mmap"
    #     if os.path.exists(rand_path):
    #         print(f"Found random mmap at {rand_path}")
    #     save_random_mmap(rand_path, args.mmap_size, args.dimension)
    #     args.dstore_fp16 = True
    #     args.mmap = rand_path
    # keys = np.memmap(
    #     args.mmap,
    #     dtype=np.float16 if args.dstore_fp16 else np.float32,
    #     mode="r",
    #     shape=(args.mmap_size, args.dimension),
    # )
    with open(args.vector_ids, 'rb') as f:
        data = pickle.load(f)

    cfg = IndexCfg(
        index_builder_type=args.index_builder_type,
        dim=args.dim,
        train_num=args.train_num,
        centroids=args.centroids,
        metric=args.metric,
        nprobe=args.nprobe,
        index_storage_dir=args.index_storage_dir,
    )

    flat_cfg = IndexCfg(
        index_builder_type='flat',
        dim=args.dim,
        train_num=args.train_num,
        centroids=args.centroids,
        metric=args.metric,
        nprobe=args.nprobe,
        index_storage_dir=args.index_storage_dir,
    )

    metric = cfg.get_metric()
    # index = faiss.IndexIVFFlat(get_quantizer(cfg), cfg.dim, cfg.centroids, metric)
    # index.nprobe = cfg.nprobe

    # index = faiss.index_factory(cfg.dim, cfg.faiss_factory)
    # metric = cfg.get_metric()
    # if metric == faiss.METRIC_INNER_PRODUCT:
    #     quantizer = faiss.IndexFlatIP(cfg.dim)
    # elif metric == faiss.METRIC_L2:
    #     quantizer = faiss.IndexFlatL2(cfg.dim)
    # else:
    #     raise RuntimeError(f"Metric={metric} is not supported for ivf_gpu factory")

    # index_ivf = faiss.extract_index_ivf(index)

    # index_ivf.clustering_index = clustering_index
    # index.nprobe = cfg.nprobe

    client = IndexClient(args.discovery_config, cfg=cfg)

    index_id = "wiki"
    flat_index_id = "wiki_flat"

    client.create_index(index_id, cfg)
    # client.create_index(flat_index_id, flat_cfg)

    index_id_to_db_id = []

    num_vec = len(data)

    for i in tqdm(list(range(0, num_vec, args.bs))):
        end = min(i + args.bs, num_vec)
        emb, id = [],[]
        for j in range(i, end):
            emb.append(data[j][1])
            id.append(data[j][0])
        emb = np.array(emb)
        index_id_to_db_id.extend(id)

        client.add_index_data(index_id, emb, id)
        client.save_index(index_id)

    with open(args.db_to_index, 'wb') as f:
        pickle.dump(index_id_to_db_id, f)

    if client.get_state(index_id) == IndexState.NOT_TRAINED:
        client.sync_train(index_id)
    while client.get_state(index_id) != IndexState.TRAINED:
        time.sleep(1)

    time0 = time.time()
    rand_vec = torch.rand((1, 768)).numpy()
    res = client.search(rand_vec, 4, index_id, return_embeddings=True)

    print("4 index search time: %f sec.", time.time() - time0)

    # client.save_index(index_id)

    print(f"ntotal: {client.get_ntotal(index_id)}")
    time0 = time.time()
    rand_vec = torch.rand(
        (
            1,
            args.dim,
        )
    ).numpy()
    print(client.search(rand_vec, 1, index_id, return_embeddings=True))
    print("50 index search time: %f sec.", time.time() - time0)


if __name__ == "__main__":
    main()
