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
    """
    Collate function for two-stage frame sampling.

    功能说明：
      1. 第一阶段：从所有帧中按顺序（可隔帧，即有规律）抽取 A 帧，
         当原始序列长度不足时，通过重复序列保证足够的采样范围。
      2. 第二阶段：从第一阶段抽取的 A 帧中随机抽取 B 帧（要求 B <= A），
         最终保证抽取出的帧保持输入的顺序。

    配置参数 sample_config 应包含：
      - frames_num_ordered: 第一阶段抽取的帧数 (A)
      - frames_num_random: 第二阶段随机抽取的帧数 (B)，必须小于等于 A
      - frames_ordered_skip: 有序抽取时隔几帧（默认 0，表示连续抽取）
      - frames_all_limit: （可选）对最终抽取的帧数上限，默认为 -1，表示不限制
    """
    def __init__(self, label_set, sample_config):
        self.label_set = label_set
        self.frames_num_ordered = sample_config['frames_num_ordered']
        self.frames_num_random = sample_config['frames_num_random']
        if self.frames_num_random > self.frames_num_ordered:
            raise ValueError("frames_num_random must be less than or equal to frames_num_ordered")
        self.frames_ordered_skip = sample_config.get('frames_ordered_skip', 0)
        self.frames_all_limit = sample_config.get('frames_all_limit', -1)

    def __call__(self, batch):
        """
        参数：
          batch: 每个元素应为一个元组 (seqs, meta)。
                 其中 seqs 是一个 list，每个元素为某一路 feature 的帧序列，
                 meta 为一个 tuple，例如 (label, type, view)。

        返回值：
          一个列表 [frames_batch, labs, types, views, extra]，
          其中 frames_batch 按 feature 分组，后续各项依次为样本标签、类型、视角及额外信息（此处为 None）。
        """
        batch_size = len(batch)
        # 每个样本可能包含多个 feature（例如 silhouette, skeleton 等），取第一路作为参考
        feature_num = len(batch[0][0])
        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []

        # 将 batch 中每个样本的数据分离存储
        for sample in batch:
            seqs, meta = sample
            seqs_batch.append(seqs)
            labs_batch.append(self.label_set.index(meta[0]))
            typs_batch.append(meta[1])
            vies_batch.append(meta[2])

        def sample_frames(seqs):
            """
            对单个样本的各 feature 序列进行两阶段抽帧：
             - 第一阶段：有序抽取 frames_num_ordered（A）帧，
               其中可根据 frames_ordered_skip 控制隔帧采样，若序列不足则通过重复保证足够的帧。
             - 第二阶段：从 A 帧中随机抽取 frames_num_random（B）帧（保持顺序）。
            """
            sampled_fras = [[] for _ in range(feature_num)]
            seq_len = len(seqs[0])
            A = self.frames_num_ordered
            B = self.frames_num_random
            skip = self.frames_ordered_skip
            # 计算所需窗口长度：间隔 (skip+1) 抽取 A 帧
            required_window = A * (skip + 1)

            # 构造初始索引列表
            index_list = list(range(seq_len))
            # 若帧数不足，则重复索引列表以覆盖所需窗口
            if seq_len < required_window:
                it = math.ceil(required_window / seq_len)
                index_list = index_list * it
            effective_length = len(index_list)

            # 随机选择起始位置以确保窗口内足够帧
            start_max = effective_length - required_window
            start = random.choice(range(start_max + 1))
            # 第一阶段：按照步长 (skip+1) 从 index_list 中有序采样 A 帧
            ordered_indices = [start + i * (skip + 1) for i in range(A)]
            # 第二阶段：从 ordered_indices 中随机选择 B 帧，并排序以保持原顺序
            selected_ordered = random.sample(ordered_indices, B)
            # 将采样得到的索引映射回实际帧索引
            actual_indices = [index_list[idx] for idx in selected_ordered]
            # 如配置中限制了最终帧数量，则做限制
            if self.frames_all_limit > -1 and len(actual_indices) > self.frames_all_limit:
                actual_indices = actual_indices[:self.frames_all_limit]
            # 对每个 feature，根据 actual_indices 取出对应的帧
            for f in range(feature_num):
                for j in actual_indices:
                    sampled_fras[f].append(seqs[f][j])
            return sampled_fras

        # 对 batch 中每个样本依次进行帧采样
        fras_batch = [sample_frames(seqs) for seqs in seqs_batch]

        # 按照原始 CollateFn 的数据组织方式按 feature 重新整理，每一路 feature 整合为 (batch_size, num_frames) 的 numpy 数组
        frames_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)] for j in range(feature_num)]

        # 返回的 batch 格式：[frames_batch, labs, types, views, extra]
        return [frames_batch, labs_batch, typs_batch, vies_batch, None]