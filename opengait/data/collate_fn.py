import math
import random
import numpy as np
from utils import get_msg_mgr


class CollateFn(object):
    def __init__(self, label_set, sample_config):
        self.label_set = label_set
        sample_type = sample_config['sample_type']
        sample_type = sample_type.split('_')
        self.sampler = sample_type[0]
        self.ordered = sample_type[1]
        if self.sampler not in ['fixed', 'unfixed', 'all']:
            raise ValueError
        if self.ordered not in ['ordered', 'unordered']:
            raise ValueError
        self.ordered = sample_type[1] == 'ordered'

        # fixed cases
        if self.sampler == 'fixed':
            self.frames_num_fixed = sample_config['frames_num_fixed']

        # unfixed cases
        if self.sampler == 'unfixed':
            self.frames_num_max = sample_config['frames_num_max']
            self.frames_num_min = sample_config['frames_num_min']

        if self.sampler != 'all' and self.ordered:
            self.frames_skip_num = sample_config['frames_skip_num']

        self.frames_all_limit = -1
        if self.sampler == 'all' and 'frames_all_limit' in sample_config:
            self.frames_all_limit = sample_config['frames_all_limit']

    def __call__(self, batch):
        batch_size = len(batch)
        # currently, the functionality of feature_num is not fully supported yet, it refers to 1 now. We are supposed to make our framework support multiple source of input data, such as silhouette, or skeleton.
        feature_num = len(batch[0][0])
        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []

        for bt in batch:
            seqs_batch.append(bt[0])
            labs_batch.append(self.label_set.index(bt[1][0]))
            typs_batch.append(bt[1][1])
            vies_batch.append(bt[1][2])

        global count
        count = 0

        def sample_frames(seqs):
            global count
            sampled_fras = [[] for i in range(feature_num)]
            seq_len = len(seqs[0])
            indices = list(range(seq_len))

            if self.sampler in ['fixed', 'unfixed']:
                if self.sampler == 'fixed':
                    frames_num = self.frames_num_fixed
                else:
                    frames_num = random.choice(
                        list(range(self.frames_num_min, self.frames_num_max+1)))

                if self.ordered:
                    fs_n = frames_num + self.frames_skip_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / seq_len)
                        seq_len = seq_len * it
                        indices = indices * it

                    start = random.choice(list(range(0, seq_len - fs_n + 1)))
                    end = start + fs_n
                    idx_lst = list(range(seq_len))
                    idx_lst = idx_lst[start:end]
                    idx_lst = sorted(np.random.choice(
                        idx_lst, frames_num, replace=False))
                    indices = [indices[i] for i in idx_lst]
                else:
                    replace = seq_len < frames_num

                    if seq_len == 0:
                        get_msg_mgr().log_debug('Find no frames in the sequence %s-%s-%s.'
                                                % (str(labs_batch[count]), str(typs_batch[count]), str(vies_batch[count])))

                    count += 1
                    indices = np.random.choice(
                        indices, frames_num, replace=replace)

            for i in range(feature_num):
                for j in indices[:self.frames_all_limit] if self.frames_all_limit > -1 and len(indices) > self.frames_all_limit else indices:
                    sampled_fras[i].append(seqs[i][j])
            return sampled_fras

        # f: feature_num
        # b: batch_size
        # p: batch_size_per_gpu
        # g: gpus_num
        fras_batch = [sample_frames(seqs) for seqs in seqs_batch]  # [b, f]
        batch = [fras_batch, labs_batch, typs_batch, vies_batch, None]

        if self.sampler == "fixed":
            fras_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                          for j in range(feature_num)]  # [f, b]
        else:
            seqL_batch = [[len(fras_batch[i][0])
                           for i in range(batch_size)]]  # [1, p]

            def my_cat(k): return np.concatenate(
                [fras_batch[i][k] for i in range(batch_size)], 0)
            fras_batch = [[my_cat(k)] for k in range(feature_num)]  # [f, g]

            batch[-1] = np.asarray(seqL_batch)

        batch[0] = fras_batch
        return batch



class CollateFnTCL(object):
    def __init__(self, label_set, sample_config):
        self.label_set = label_set
        self.frames_num_ordered = sample_config['frames_num_ordered']     # A
        self.frames_ordered_skip = sample_config.get('frames_ordered_skip', 0)
        self.frames_all_limit = sample_config.get('frames_all_limit', -1)
        self.ordered = sample_config.get('ordered', False)
        
        # 新增：是否使用 batch-level 随机帧数范围
        self.sample_random_range = sample_config.get('sample_random_range', None)

        # 如果使用固定 B 值，也保留旧参数逻辑
        self.frames_num_random = sample_config.get('frames_num_random', None)
        if self.frames_num_random is not None and self.sample_random_range is not None:
            raise ValueError("Use either 'frames_num_random' (fixed) or 'sample_random_range' (variable), not both.")

    def __call__(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []

        for sample in batch:
            seqs, meta = sample
            seqs_batch.append(seqs)
            labs_batch.append(self.label_set.index(meta[0]))
            typs_batch.append(meta[1])
            vies_batch.append(meta[2])

        # ⭐ 生成本 batch 的随机 B 值（第二阶段采样帧数）
        if self.sample_random_range is not None:
            B = random.randint(self.sample_random_range[0], self.sample_random_range[1])
        else:
            B = self.frames_num_random
        if B > self.frames_num_ordered:
            raise ValueError("Sampled B must be ≤ frames_num_ordered")

        def sample_frames(seqs):
            sampled_fras = [[] for _ in range(feature_num)]
            seq_len = len(seqs[0])
            A = self.frames_num_ordered
            skip = self.frames_ordered_skip
            required_window = A * (skip + 1)
            index_list = list(range(seq_len))

            if seq_len < required_window:
                it = math.ceil(required_window / seq_len)
                index_list = index_list * it
            effective_length = len(index_list)
            start_max = effective_length - required_window
            start = random.choice(range(start_max + 1))

            ordered_indices = [start + i * (skip + 1) for i in range(A)]
            selected_ordered = random.sample(ordered_indices, B)
            if self.ordered:
                selected_ordered.sort()

            actual_indices = [index_list[idx] for idx in selected_ordered]
            if self.frames_all_limit > -1 and len(actual_indices) > self.frames_all_limit:
                actual_indices = actual_indices[:self.frames_all_limit]

            for f in range(feature_num):
                for j in actual_indices:
                    sampled_fras[f].append(seqs[f][j])
            return sampled_fras

        fras_batch = [sample_frames(seqs) for seqs in seqs_batch]
        frames_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)] for j in range(feature_num)]

        return [frames_batch, labs_batch, typs_batch, vies_batch, None]