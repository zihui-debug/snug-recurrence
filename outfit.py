import tensorflow as tf
import numpy as np

def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    print(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'
          .format(total, used, free))
    return total, used, free

class Outfit:
    def __init__(self, v_cloth, f_cloth):
        self._T,self._F = v_cloth, f_cloth
        self._outfit_info()


    def _outfit_info(self):
        self._E = self.faces2edges(self._F)
        self._neigh_F = self.neigh_faces(self._F, self._E)  # edges of graph representing face connectivity
        self._precompute_area()
        self._precompute_edges()

    def faces2edges(self, F):
        E = set()
        for f in F:
            N = len(f)
            for i in range(N):
                j = (i + 1) % N
                E.add(tuple(sorted([f[i], f[j]])))
        return np.array(list(E), np.int32)

    def edges2graph(self, E):
        G = {}
        for e in E:
            if not e[0] in G: G[e[0]] = {}
            if not e[1] in G: G[e[1]] = {}
            G[e[0]][e[1]] = 1
            G[e[1]][e[0]] = 1
        return G

    def neigh_faces(self, F, E=None):
        if E is None: E = self.faces2edges(F)
        G = {tuple(e): [] for e in E}
        for i, f in enumerate(F):
            n = len(f)
            for j in range(n):
                k = (j + 1) % n
                e = tuple(sorted([f[j], f[k]]))
                G[e] += [i]
        neighF = []
        for key in G:
            if len(G[key]) == 2:
                neighF += [G[key]]
            elif len(G[key]) > 2:
                print("Neigh F unexpected behaviour")
                continue
        return np.array(neighF, np.int32)

    def _precompute_edges(self):
        T, E = self._T, self._E
        e = tf.gather(T, E[:, 0], axis=0) - tf.gather(T, E[:, 1], axis=0)
        self._edges = tf.sqrt(tf.reduce_sum(e ** 2, -1))

    def _precompute_area(self):
        T, F = self._T, self._F
        u = tf.gather(T, F[:, 2], axis=0) - tf.gather(T, F[:, 0], axis=0)
        v = tf.gather(T, F[:, 1], axis=0) - tf.gather(T, F[:, 0], axis=0)
        areas = tf.norm(tf.linalg.cross(u, v), axis=-1)
        self._total_area = tf.reduce_sum(areas) / 2.0
        # print("Total cloth area: ", self._total_area.numpy())