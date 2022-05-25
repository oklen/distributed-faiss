import faiss
import numpy as np

res = faiss.StandardGpuResources()

d = 768

quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, 768, faiss.METRIC_INNER_PRODUCT)
index.nprobe = 384

index = faiss.index_cpu_to_gpu(res, 0, index)

nb = 100000
nq = 10000
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000

index.train(xb)
quantizer.train(xb)

index.add(xb)
quantizer.add(xb)


scores, ids = index.search(xq, 5)
scores, ids, vectors = quantizer.search_and_reconstruct(xq, 1)
print(type(ids))
print(len(ids))
print(len(ids[0]))
print(ids[0][0])
print(type(ids[0][0]))
import time
print(type(vectors))
print(type(vectors[0]))

t = time.time()
embds = list(map(lambda x: list(map(lambda q:quantizer.reconstruct(int(q)), x)), ids))
embds = np.array(embds)
# print(embds)
# embds = list(map(lambda x: list(map(quantizer.reconstruct, x)), ids))
print(time.time()-t)
# print(embds[0][0])
# embds = [quantizer.reconstruct(int(i)) for q in ids for i in q]
print(embds.shape)
# print(quantizer.reconstruct(0))