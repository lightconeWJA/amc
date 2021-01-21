# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import time
import torch
import torch.nn as nn
from lib.utils import AverageMeter, accuracy, prGreen
from lib.data import get_split_dataset
from env.rewards import *
import math

import numpy as np
import copy


class ChannelPruningEnv:
    """
    Env for channel pruning search
    """
    def __init__(self, model, checkpoint, data, preserve_ratio, args, n_data_worker=4,
                 batch_size=256, export_model=False, use_new_input=False):
        # default setting
        self.prunable_layer_types = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]

        # save options
        self.model = model
        self.checkpoint = checkpoint
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data
        self.preserve_ratio = preserve_ratio

        # options from args
        self.args = args
        self.lbound = args.lbound
        self.rbound = args.rbound

        self.use_real_val = args.use_real_val

        self.n_calibration_batches = args.n_calibration_batches
        self.n_points_per_layer = args.n_points_per_layer
        self.channel_round = args.channel_round
        self.acc_metric = args.acc_metric
        self.data_root = args.data_root

        self.export_model = export_model
        self.use_new_input = use_new_input

        # sanity check
        assert self.preserve_ratio > self.lbound, 'Error! You can make achieve preserve_ratio smaller than lbound!'

        # prepare data
        self._init_data()

        # build indexs
        # 初始化可剪通道的网络模块对应索引的集合
        # 同时初始化buffer_idx，即所有depthwise层对应索引的集合
        self._build_index()

        # 此变量表示可剪通道的网络层的数目
        self.n_prunable_layer = len(self.prunable_idx)

        # extract information for preparing
        # 针对每个prunable layer进行信息记录，输入输出记录/采样
        self._extract_layer_information()

        # build embedding (static part)
        self._build_state_embedding()

        # build reward
        self.reset()  # restore weight
        self.org_acc = self._validate(self.val_loader, self.model)
        print('=> original acc: {:.3f}%'.format(self.org_acc))
        self.org_model_size = sum(self.wsize_list)
        print('=> original weight size: {:.4f} M param'.format(self.org_model_size * 1. / 1e6))
        self.org_flops = sum(self.flops_list)
        print('=> FLOPs:')
        print([self.layer_info_dict[idx]['flops']/1e6 for idx in sorted(self.layer_info_dict.keys())])
        print('=> original FLOPs: {:.4f} M'.format(self.org_flops * 1. / 1e6))

        self.expected_preserve_computation = self.preserve_ratio * self.org_flops

        self.reward = eval(args.reward)

        # 初始化：最优reward为负无穷，最优策略为空，最优维度为空
        self.best_reward = -math.inf
        self.best_strategy = None
        self.best_d_prime_list = None

        # 初始化参数量，之前算过
        self.org_w_size = sum(self.wsize_list)

    # 根据DDPG agent所选择的action进行一次模拟剪枝，并记录剪枝后的reward
    # 并不会真的剪枝，记录好reward后便会恢复模型，只有在最后一次episode时候才会进行真正的剪枝
    # 第一次使用此函数的上下文：
    # self.cur_ind 在 resset() 中置为0
    # self.strategy_dict 在_build_index()中变为{2: [1, 0.2], ... , 52:[0.2, 0.2], ..., 98:[0.2, 1]}形式
    # self.prunable_idx 在 _build_index里已经存好了所有可prunable layer的index
    # self.index_buffer 空的
    def step(self, action):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        if self.visited[self.cur_ind]:
            action = self.strategy_dict[self.prunable_idx[self.cur_ind]][0]
            preserve_idx = self.index_buffer[self.cur_ind]
        else:
            action = self._action_wall(action)  # percentage to preserve
            preserve_idx = None

        # prune and update action
        action, d_prime, preserve_idx = self.prune_kernel(self.prunable_idx[self.cur_ind], action, preserve_idx)

        if not self.visited[self.cur_ind]:
            for group in self.shared_idx:
                if self.cur_ind in group:  # set the shared ones
                    for g_idx in group:
                        self.strategy_dict[self.prunable_idx[g_idx]][0] = action
                        self.strategy_dict[self.prunable_idx[g_idx - 1]][1] = action
                        self.visited[g_idx] = True
                        self.index_buffer[g_idx] = preserve_idx.copy()

        if self.export_model:  # export checkpoint
            print('# Pruning {}: ratio: {}, d_prime: {}'.format(self.cur_ind, action, d_prime))

        self.strategy.append(action)  # save action to strategy
        self.d_prime_list.append(d_prime)

        self.strategy_dict[self.prunable_idx[self.cur_ind]][0] = action
        if self.cur_ind > 0:
            self.strategy_dict[self.prunable_idx[self.cur_ind - 1]][1] = action

        # all the actions are made
        if self._is_final_layer():
            assert len(self.strategy) == len(self.prunable_idx)
            current_flops = self._cur_flops()
            acc_t1 = time.time()
            acc = self._validate(self.val_loader, self.model)
            acc_t2 = time.time()
            self.val_time = acc_t2 - acc_t1
            compress_ratio = current_flops * 1. / self.org_flops
            info_set = {'compress_ratio': compress_ratio, 'accuracy': acc, 'strategy': self.strategy.copy()}
            reward = self.reward(self, acc, current_flops)

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy.copy()
                self.best_d_prime_list = self.d_prime_list.copy()
                prGreen('New best reward: {:.4f}, acc: {:.4f}, compress: {:.4f}'.format(self.best_reward, acc, compress_ratio))
                prGreen('New best policy: {}'.format(self.best_strategy))
                prGreen('New best d primes: {}'.format(self.best_d_prime_list))

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            if self.export_model:  # export state dict
                torch.save(self.model.state_dict(), self.export_path)
                return None, None, None, None
            return obs, reward, done, info_set

        info_set = None
        reward = 0
        done = False
        self.visited[self.cur_ind] = True  # set to visited
        self.cur_ind += 1  # the index of next layer
        # build next state (in-place modify)
        self.layer_embedding[self.cur_ind][-3] = self._cur_reduced() * 1. / self.org_flops  # reduced
        self.layer_embedding[self.cur_ind][-2] = sum(self.flops_list[self.cur_ind + 1:]) * 1. / self.org_flops  # rest
        self.layer_embedding[self.cur_ind][-1] = self.strategy[-1]  # last action
        obs = self.layer_embedding[self.cur_ind, :].copy()

        return obs, reward, done, info_set


    def reset(self):
        # restore env by loading the checkpoint
        self.model.load_state_dict(self.checkpoint)
        self.cur_ind = 0
        self.strategy = []  # pruning strategy
        self.d_prime_list = []
        self.strategy_dict = copy.deepcopy(self.min_strategy_dict)
        # reset layer embeddings
        self.layer_embedding[:, -1] = 1.
        self.layer_embedding[:, -2] = 0.
        self.layer_embedding[:, -3] = 0.
        obs = self.layer_embedding[0].copy()
        obs[-2] = sum(self.wsize_list[1:]) * 1. / sum(self.wsize_list)
        self.extract_time = 0
        self.fit_time = 0
        self.val_time = 0
        # for share index
        self.visited = [False] * len(self.prunable_idx)
        self.index_buffer = {}
        return obs

    def set_export_path(self, path):
        self.export_path = path

    
    # action, d_prime, preserve_idx = self.prune_kernel(self.prunable_idx[self.cur_ind], action, preserve_idx)
    # op_idx (operation index) <-> self.prunable_idx[self.cur_ind]：调用此func时候处理的prunable layer的index
    # action <-> preserve_ratio：actor网络输出的或者随机选取的压缩比
    # preserve_idx <-> preserve_idx：
    def prune_kernel(self, op_idx, preserve_ratio, preserve_idx=None):
        '''Return the real ratio'''
        # 取得 operation index 所指代的需要裁剪的 layer，此函数中用 op 引用
        m_list = list(self.model.modules())
        op = m_list[op_idx]
        
        assert (preserve_ratio <= 1.)

        if preserve_ratio == 1:  # do not prune
            return 1., op.weight.size(1), None  # TODO: should be a full index
            # n, c, h, w = op.weight.size()
            # mask = np.ones([c], dtype=bool)

        def format_rank(x):
            rank = int(np.around(x)) # 四舍五入
            return max(rank, 1)

        # 例如一个 op 为 Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # 那么 op.weight.shape = [64, 32, 1, 1] 即为 [N, C, H, W]
        n, c = op.weight.size(0), op.weight.size(1)

        # 计算压缩后 卷积核的channel数量，并保证了一定大于1
        d_prime = format_rank(c * preserve_ratio)

        # np.ceil() 向上取整, np.floor() 向下取整
        # 即把 d_prime 搞成channel_round的整数倍，且保证d_prime <= c
        d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)
        if d_prime > c:
            d_prime = int(np.floor(c * 1. / self.channel_round) * self.channel_round)

        # 提取信息
        # X：op_idx对应layer的input_feat
        # Y：op_idx对应layer的output_feat
        # weight：op_idx对应layer的weight，用于挑选需要剪的通道
        extract_t1 = time.time()
        if self.use_new_input:  # this is slow and may lead to overfitting
            self._regenerate_input_feature()
        X = self.layer_info_dict[op_idx]['input_feat']  # input after pruning of previous ops
        Y = self.layer_info_dict[op_idx]['output_feat']  # fixed output from original model
        weight = op.weight.data.cpu().numpy()
        # conv [C_out, C_in, ksize, ksize]
        # fc [C_out, C_in]
        op_type = 'Conv2D'
        if len(weight.shape) == 2:
            op_type = 'Linear'
            weight = weight[:, :, None, None] # [out, in] 变成了 [out, in, 1, 1]
        extract_t2 = time.time()
        self.extract_time += extract_t2 - extract_t1
        fit_t1 = time.time()

        # preserve_idx 在 self.visited[self.cur_ind] == 0 时候 为 None
        # 沿着维度C_in(输入通道数)进行sum操作
        # np.argsort()会进行排序后返回排序的下标结果
        # 这部分就是把weight沿着sum相加后最小的一部分channel给去掉
        # 因为是 取绝对值->取反->排序，能保证去掉的都是绝对值的和最小的一部分
        # 得到的preserve_idx就是拟剪枝后保留的这些通道
        if preserve_idx is None:  # not provided, generate new
            importance = np.abs(weight).sum((0, 2, 3))
            sorted_idx = np.argsort(-importance)  # sum magnitude along C_in, sort descend
            preserve_idx = sorted_idx[:d_prime]  # to preserve index
        assert len(preserve_idx) == d_prime
        
        # 根据保留的通道设定mask数组，一个通道对应一个元素
        # 其中保留的部分设定为True，要剪的设定为false
        mask = np.zeros(weight.shape[1], bool)
        mask[preserve_idx] = True

        # reconstruct, X, Y <= [N, C]
        # 进行剪枝后使用线性回归对参数进行重新调整
        # 根据之前new_forward()所记录的'input_feat' 与 'output_feat' 
        # 来对模拟剪通道以后的layer进行初步调整来尽量模拟之前的输入输出的函数
        # conv情况 [C_out, C_in, ksize, ksize]，ksize == 1
        # fc情况 [C_out, C_in, 1, 1]
        masked_X = X[:, mask]
        if weight.shape[2] == 1:  # 1x1 conv or fc
            from lib.utils import least_square_sklearn
            rec_weight = least_square_sklearn(X=masked_X, Y=Y)
            rec_weight = rec_weight.reshape(-1, 1, 1, d_prime)  # (C_out, K_h, K_w, C_in')
            rec_weight = np.transpose(rec_weight, (0, 3, 1, 2))  # (C_out, C_in', K_h, K_w)
        else:
            raise NotImplementedError('Current code only supports 1x1 conv now!')
        # 一个episode走下来后每层对应的rec_weight.shape差不多是这个鸟样：
        # (64, 8, 1, 1)
        # (128, 24, 1, 1)
        # (128, 112, 1, 1)
        # (256, 104, 1, 1)
        # (256, 112, 1, 1)
        # (512, 112, 1, 1)
        # (512, 104, 1, 1)
        # (512, 216, 1, 1)
        # (512, 440, 1, 1)
        # (512, 296, 1, 1)
        # (512, 216, 1, 1)
        # (1024, 144, 1, 1)
        # (1024, 848, 1, 1)
        # (1000, 944, 1, 1)
        
        # 参数默认情况下会执行
        # 相当于将需要裁剪的所有通道的权重变为0
        if not self.export_model:  # pad, pseudo compress
            rec_weight_pad = np.zeros_like(weight) # 搞一个和weight相同shape的全0矩阵
            rec_weight_pad[:, mask, :, :] = rec_weight
            rec_weight = rec_weight_pad

        # 针对fc层重新变回正确的shape
        if op_type == 'Linear':
            rec_weight = rec_weight.squeeze()
            assert len(rec_weight.shape) == 2
        
        fit_t2 = time.time()
        self.fit_time += fit_t2 - fit_t1
        # now assign
        op.weight.data = torch.from_numpy(rec_weight).cuda()
        action = np.sum(mask) * 1. / len(mask)  # calculate the ratio

        # 这部分暂时不用看
        if self.export_model:  # prune previous buffer ops
            prev_idx = self.prunable_idx[self.prunable_idx.index(op_idx) - 1]
            for idx in range(prev_idx, op_idx):
                m = m_list[idx]
                if type(m) == nn.Conv2d:  # depthwise
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask, :, :, :]).cuda()
                    if m.groups == m.in_channels:
                        m.groups = int(np.sum(mask))
                elif type(m) == nn.BatchNorm2d:
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask]).cuda()
                    m.bias.data = torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda()
                    m.running_mean.data = torch.from_numpy(m.running_mean.data.cpu().numpy()[mask]).cuda()
                    m.running_var.data = torch.from_numpy(m.running_var.data.cpu().numpy()[mask]).cuda()
        
        return action, d_prime, preserve_idx

    def _is_final_layer(self):
        return self.cur_ind == len(self.prunable_idx) - 1

    def _action_wall(self, action):
        # self.startegy 负责储存每个episode中每一个prunable layer的preserve ratio
        # 因此这个列表的长度应该和 cur_ind应该同时更新并保持一致
        assert len(self.strategy) == self.cur_ind

        action = float(action)
        action = np.clip(action, 0, 1)

        # this_comp（this prunable layer compression） 为遍历中的 prunable layer 的flops
        # other_comp 为除了this_comp 对应的层外其他的层的flops之和
        other_comp = 0
        this_comp = 0

        # 这里self.layer_info_dict[idx]['flops'] 在_extract_layer_information()定义
        # 并且其中包含了所有的 prunable_idx 与 buffer_idx 所有层的信息
        # 包括 input_feat, output_feat, randx, randy, params, flops
        # _extract_layer_information() 调用 new_forward()
        # new_forward() 调用 measure_layer_for_pruning 对每个layer增加了flops与param两个成员
        for i, idx in enumerate(self.prunable_idx):
            flop = self.layer_info_dict[idx]['flops']
            buffer_flop = self._get_buffer_flops(idx)

            # 针对不同layer所处位置的不同情况计算 this_comp 和 other_comp
            if i == self.cur_ind - 1:  # TODO: add other member in the set
                this_comp += flop * self.strategy_dict[idx][0]
                # add buffer (but not influenced by ratio)
                other_comp += buffer_flop * self.strategy_dict[idx][0]
            elif i == self.cur_ind:
                this_comp += flop * self.strategy_dict[idx][1]
                # also add buffer here (influenced by ratio)
                this_comp += buffer_flop
            else:
                other_comp += flop * self.strategy_dict[idx][0] * self.strategy_dict[idx][1]
                # add buffer
                other_comp += buffer_flop * self.strategy_dict[idx][0]  # only consider input reduction

        self.expected_min_preserve = other_comp + this_comp * action
        max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp

        action = np.minimum(action, max_preserve_ratio)
        action = np.maximum(action, self.strategy_dict[self.prunable_idx[self.cur_ind]][0])  # impossible (should be)

        return action

    # 从buffer_dict中取得某个prunable layer的所有buffer layer
    # 之后对所有的buffer layer 的 flops 求和并返回
    def _get_buffer_flops(self, idx):
        buffer_idx = self.buffer_dict[idx]
        buffer_flop = sum([self.layer_info_dict[_]['flops'] for _ in buffer_idx])
        return buffer_flop

    def _cur_flops(self):
        flops = 0
        for i, idx in enumerate(self.prunable_idx):
            c, n = self.strategy_dict[idx]  # input, output pruning ratio
            flops += self.layer_info_dict[idx]['flops'] * c * n
            # add buffer computation
            flops += self._get_buffer_flops(idx) * c  # only related to input channel reduction
        return flops

    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_flops - self._cur_flops()
        return reduced

    def _init_data(self):
        # split the train set into train + val
        # for CIFAR, split 5k for val
        # for ImageNet, split 3k for val
        val_size = 5000 if 'cifar' in self.data_type else 3000
        self.train_loader, self.val_loader, n_class = get_split_dataset(self.data_type, self.batch_size,
                                                                        self.n_data_worker, val_size,
                                                                        data_root=self.data_root,
                                                                        use_real_val=self.use_real_val,
                                                                        shuffle=False)  # same sampling
        if self.use_real_val:  # use the real val set for eval, which is actually wrong
            print('*** USE REAL VALIDATION SET!')

    # 定义一堆杂七杂八的dict，用于之后的剪通道操作
    # strategy_dict 作为一个字典
    # 以prunalbe layer 的 index 作为 key
    # 以prunable layer 的输入输出的“保留值”（即裁剪后还留多少）的list[输入保留， 输出保留] 作为value
    def _build_index(self):
        self.prunable_idx = []      # 记录可剪枝网络模块所对应的索引号
        self.prunable_ops = []      # 没用，记录prunable_idx对应的网络模块
        self.layer_type_dict = {}   # 没用，可看作是上面两个变量合起来后的字典
        self.strategy_dict = {}     
        self.buffer_dict = {}
        this_buffer_list = []
        self.org_channels = []

        # build index and the min strategy dict
        # modules()会采用bfs(广度优先，即逐层遍历)方式返回所有网络的迭代器
        # 结果中会有网络本身，网络的子模块，子模块的子模块......
        # 和bfs略有不同的是，modules()只返回“模块级”的部分
        # 意思是如果有完全相同的conv2D或者Linear，则只返回一次
        # 此循环的任务就是找到最底层的子模块(可以理解为叶子节点)中只包含一个卷积层或全连接层的部分，找到后进行记录
        # i 就是 idx，代表不同模块的编号
        # m 就是不同的网络模块
        for i, m in enumerate(self.model.modules()):
            # 本模型只负责2d卷积以及全连接
            # 从prunable_layer_types定义可看出返回网络的种类定义规则
            if type(m) in self.prunable_layer_types:
                # depth-wise(深度分离)卷积
                # 当 gruops == in_channels，意味着每个输入channel都对应单独的一组滤波器
                # 这时候就是深度分离卷积的情况，会将其单独放在buffer_idx中
                # 针对其余卷积层与全连接层，会将其放入prunable_idx中，表示可剪通道
                # 进行遍历的过程中，每遇到一个深度卷积就放进this_buffer_list中暂存
                # 遍历到普通卷积或者全连接后，就将this_buffer_list中暂存的都作为此prunable layer的buffer
                # 之后清空this_buffer_list
                if type(m) == nn.Conv2d and m.groups == m.in_channels:  # depth-wise conv, buffer
                    this_buffer_list.append(i)
                else:  # really prunable
                    self.prunable_idx.append(i)
                    self.prunable_ops.append(m)
                    self.layer_type_dict[i] = type(m)
                    self.buffer_dict[i] = this_buffer_list
                    # 清空this_buffer_list同时并不会影响buffer_dict()
                    # 这时候咋又不出deep_copy之类的幺蛾子了......
                    this_buffer_list = []  # empty
                    self.org_channels.append(m.in_channels if type(m) == nn.Conv2d else m.in_features)

                    self.strategy_dict[i] = [self.lbound, self.lbound]

        self.strategy_dict[self.prunable_idx[0]][0] = 1  # modify the input
        self.strategy_dict[self.prunable_idx[-1]][1] = 1  # modify the output

        self.shared_idx = []
        if self.args.model == 'mobilenetv2':  # TODO: to be tested! Share index for residual connection
            connected_idx = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]  # to be partitioned
            last_ch = -1
            share_group = None
            for c_idx in connected_idx:
                if self.prunable_ops[c_idx].in_channels != last_ch:  # new group
                    last_ch = self.prunable_ops[c_idx].in_channels
                    if share_group is not None:
                        self.shared_idx.append(share_group)
                    share_group = [c_idx]
                else:  # same group
                    share_group.append(c_idx)
            print('=> Conv layers to share channels: {}'.format(self.shared_idx))

        self.min_strategy_dict = copy.deepcopy(self.strategy_dict)

        self.buffer_idx = []
        for k, v in self.buffer_dict.items():
            self.buffer_idx += v

        print('=> Prunable layer idx: {}'.format(self.prunable_idx))
        print('=> Buffer layer idx: {}'.format(self.buffer_idx))
        print('=> Initial min strategy dict: {}'.format(self.min_strategy_dict))

        # added for supporting residual connections during pruning
        # 将所有可剪通道对应的网络模块ID的visited数组置为false，完成初始化
        self.visited = [False] * len(self.prunable_idx)
        self.index_buffer = {}

    def _extract_layer_information(self):
        m_list = list(self.model.modules())

        self.data_saver = []
        self.layer_info_dict = dict()
        self.wsize_list = []
        self.flops_list = []

        from lib.utils import measure_layer_for_pruning

        # extend the forward fn to record layer info
        # 使用新成员 old_forward 来存放原来的forward函数
        # 使用新成员 new_forward 来存放扩展过功能的forward函数
        # 这里搞了个 new_forward 这么复杂的函数套函数是为了实现闭包
        # 因为在改写forward过程中，必然会涉及到对新的forward外部作用域内变量的引用
        # 上面一行所说的外部作用域的变量包括layer本身的成员
        # 闭包每次运行是能记住引用的外部作用域的变量的值，正式我们需要的
        def new_forward(m):
            def lambda_forward(x):
                m.input_feat = x.clone()
                measure_layer_for_pruning(m, x)
                y = m.old_forward(x)
                m.output_feat = y.clone()
                return y

            return lambda_forward

        # 改写所有的conv2d与linear层的forward函数
        for idx in self.prunable_idx + self.buffer_idx:  # get all
            m = m_list[idx]
            m.old_forward = m.forward
            m.forward = new_forward(m)

        print('=> Extracting information...')
        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.train_loader):
                if i_b == self.n_calibration_batches:
                    break
                self.data_saver.append((input.clone(), target.clone()))
                input_var = torch.autograd.Variable(input).cuda()

                # inference and collect stats
                # 就是用input_var来过一遍模型 
                # 并不关心输出
                _ = self.model(input_var)

                # 第一个batch的操作：
                # 将layer_info_dict搞成一个二维dict
                # 里面存prunable和buffer的参数量与计算量两个信息 
                # 并用wsize_list和flops_list 分开再存一次
                if i_b == 0:  # first batch
                    for idx in self.prunable_idx + self.buffer_idx:
                        self.layer_info_dict[idx] = dict()
                        self.layer_info_dict[idx]['params'] = m_list[idx].params
                        self.layer_info_dict[idx]['flops'] = m_list[idx].flops
                        self.wsize_list.append(m_list[idx].params)
                        self.flops_list.append(m_list[idx].flops)
                
                # 针对每个batch推理完成后，每一个prunable layer进行操作
                for idx in self.prunable_idx:
                    # 类型转换，就是提取出input feature与output feature
                    # 这两个变量就是input与output的clone()结果
                    f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                    f_out_np = m_list[idx].output_feat.data.cpu().numpy()

                    # 针对当前层为卷积的情况
                    # 对于第一个卷积层不做记录，因为第一层需要保留原通道保证正常输入
                    # 对于“通常”的卷积层，直接存进layer_info_dict
                    if len(f_in_np.shape) == 4:  # conv
                        if self.prunable_idx.index(idx) == 0:  # first conv
                            f_in2save, f_out2save = None, None
                        elif m_list[idx].weight.size(3) > 1:  # normal conv
                            f_in2save, f_out2save = f_in_np, f_out_np
                        else:  
                            # 1x1 conv
                            # assert f_out_np.shape[2] == f_in_np.shape[2]  # now support k=3
                            # input格式: [N, C, H, W]
                            # 针对1*1卷积进行的则是采样处理(总体数据量太大？)
                            # n_points_per_layer的含义就是针对每个batch的每个特征图进行采样点的数量
                            # randx与randy可以理解成每个采样点的横纵坐标
                            randx = np.random.randint(0, f_out_np.shape[2] - 0, self.n_points_per_layer)
                            randy = np.random.randint(0, f_out_np.shape[3] - 0, self.n_points_per_layer)

                            # 顺便记录每个采样点的位置(我猜这个之后也不会再用)
                            self.layer_info_dict[idx][(i_b, 'randx')] = randx.copy()
                            self.layer_info_dict[idx][(i_b, 'randy')] = randy.copy()

                            # 将每个batch摊平
                            # 例如之前是[40, 32, 112, 112]，分别对应[N, C, H, W]
                            # 采样后就会变成[40, 32, 10]，10 对应的就是 112*112 特征图中的采样点
                            # 当然在剪mobilenetV1过程中，剪的是 kernel = 1*1
                            # 最后存储格式为[400, 32]，batchsize与采样数量合并
                            f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1)\
                                .reshape(self.batch_size * self.n_points_per_layer, -1)

                            f_out2save = f_out_np[:, :, randx, randy].copy().transpose(0, 2, 1) \
                                .reshape(self.batch_size * self.n_points_per_layer, -1)
                    else:
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np.copy()
                        f_out2save = f_out_np.copy()
                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                        self.layer_info_dict[idx]['output_feat'] = f_out2save
                    else:
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save))
                        self.layer_info_dict[idx]['output_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['output_feat'], f_out2save))


    def _regenerate_input_feature(self):
        # only re-generate the input feature
        m_list = list(self.model.modules())

        # delete old features
        for k, v in self.layer_info_dict.items():
            if 'input_feat' in v:
                v.pop('input_feat')

        # now let the image flow
        print('=> Regenerate features...')

        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.data_saver):
                input_var = torch.autograd.Variable(input).cuda()

                # inference and collect stats
                _ = self.model(input_var)

                for idx in self.prunable_idx:
                    f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                    if len(f_in_np.shape) == 4:  # conv
                        if self.prunable_idx.index(idx) == 0:  # first conv
                            f_in2save = None
                        else:
                            randx = self.layer_info_dict[idx][(i_b, 'randx')]
                            randy = self.layer_info_dict[idx][(i_b, 'randy')]
                            f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1)\
                                .reshape(self.batch_size * self.n_points_per_layer, -1)
                    else:  # fc
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np.copy()
                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                    else:
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save))

    # 顾名思义，我猜是确定强化学习中statues初始化相关的部分
    # embedding格式：
    # [
    #   prunable_layer的index,
    #   layer的类型(0是卷积，1是全连接)，
    #   输入大小(in_features是全连接输入大小，in_channels是卷积输入通道数)，
    #   输出大小(参考上面)，
    #   每层的步长stride(当然全连接层没有，设定为0)，
    #   每层的kernel_size(全连接层设定为1)，
    #   本层的压缩率，
    #   剩余的压缩率指标，
    #   上一层的压缩率
    # ]    
    def _build_state_embedding(self):
        # build the static part of the state embedding
        # 函数内的临时变量，函数结束时会放进类成员中
        layer_embedding = []

        # 又是对prunable layer的遍历过程
        module_list = list(self.model.modules())
        for i, ind in enumerate(self.prunable_idx):
            m = module_list[ind]
            this_state = []
            if type(m) == nn.Conv2d:
                this_state.append(i)  # index
                this_state.append(0)  # layer type, 0 for conv
                this_state.append(m.in_channels)  # in channels
                this_state.append(m.out_channels)  # out channels
                this_state.append(m.stride[0])  # stride
                this_state.append(m.kernel_size[0])  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size
            elif type(m) == nn.Linear:
                this_state.append(i)  # index
                this_state.append(1)  # layer type, 1 for fc
                this_state.append(m.in_features)  # in channels
                this_state.append(m.out_features)  # out channels
                this_state.append(0)  # stride
                this_state.append(1)  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size

            # this 3 features need to be changed later
            this_state.append(0.)  # reduced
            this_state.append(0.)  # rest
            this_state.append(1.)  # a_{t-1}
            layer_embedding.append(np.array(this_state))

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding

    def _validate(self, val_loader, model, verbose=False):
        '''
        Validate the performance on validation set
        :param val_loader:
        :param model:
        :param verbose:
        :return:
        '''
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        criterion = nn.CrossEntropyLoss().cuda()
        # switch to evaluate mode
        model.eval()
        end = time.time()

        t1 = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                input_var = torch.autograd.Variable(input).cuda()
                target_var = torch.autograd.Variable(target).cuda()

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f    top1: %.3f    top5: %.3f    time: %.3f' %
                  (losses.avg, top1.avg, top5.avg, t2 - t1))
        if self.acc_metric == 'acc1':
            return top1.avg
        elif self.acc_metric == 'acc5':
            return top5.avg
        else:
            raise NotImplementedError
