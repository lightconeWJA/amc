# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from lib.memory import SequentialMemory
from lib.utils import to_numpy, to_tensor

criterion = nn.MSELoss()
USE_CUDA = torch.cuda.is_available()


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


class Critic(nn.Module):
    # 定义两个输入层，即fcl1与fcl2
    # 这两层会同时输入到下一层
    # 因此本critic中直接通过一个全连接层变成一个尺寸后加上去了
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        self.fc11 = nn.Linear(nb_states, hidden1)
        self.fc12 = nn.Linear(nb_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, xs):
        x, a = xs
        out = self.fc11(x) + self.fc12(a)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class DDPG(object):
    # nb_states 就是论文中定义的每层的state那包括一长串所对应的向量大小
    # nb_actions 就是输出action所对应的向量大小，因为只有一个稀疏率，因此只为1
    # args.hidden1 就是nn.Linear in_feartures参数，即输入张量的大小
    # args.hidden2 就是out_features，输出张量大小
    def __init__(self, nb_states, nb_actions, args):

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            # 'init_w': args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_a)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_c)

        self.hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        self.hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        # self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu,
        #                                                sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon
        self.lbound = 0.  # args.lbound
        self.rbound = 1.  # args.rbound

        # noise
        self.init_delta = args.init_delta
        self.delta_decay = args.delta_decay
        self.warmup = args.warmup

        #
        self.epsilon = 1.0
        # self.s_t = None  # Most recent state
        # self.a_t = None  # Most recent action
        self.is_training = True

        #
        if USE_CUDA: self.cuda()

        # moving average baseline
        self.moving_average = None
        self.moving_alpha = 0.5  # based on batch, so small

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # normalize the reward
        batch_mean_reward = np.mean(reward_batch)
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
        reward_batch -= self.moving_average
        # if reward_batch.std() > 0:
        #     reward_batch /= reward_batch.std()

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                to_tensor(next_state_batch),
                self.actor_target(to_tensor(next_state_batch)),
            ])

        target_q_batch = to_tensor(reward_batch) + \
                         self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])


        # MSELoss
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t, s_t1, a_t, done):
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, done)  # save to memory
            # self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(self.lbound, self.rbound, self.nb_actions)
        # self.a_t = action
        return action

    # s_t 就是网络中一个layer中的state_embedding
    # 因此s_t可以作为actor网络的输入
    def select_action(self, s_t, episode):
        # assert episode >= self.warmup, 'Episode: {} warmup: {}'.format(episode, self.warmup)
        # squeeze()可以删除但唯独条目，即把shape为1的维度去掉
        action = to_numpy(self.actor(to_tensor(np.array(s_t).reshape(1, -1)))).squeeze(0)

        # 通过默认值为0.95的衰减率求得目前的delta作为截断正态分布的方差
        delta = self.init_delta * (self.delta_decay ** (episode - self.warmup))
        # action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        # 根据action与delta得到一个截断正态分布的采样，相当于有加入了一定的随机性
        action = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=action, sigma=delta)
        # 没必要吧，毕竟上一句已经能够限定范围了
        action = np.clip(action, self.lbound, self.rbound)

        # self.a_t = action
        return action

    def reset(self, obs):
        pass
        # self.s_t = obs
        # self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


    # truncnorm.rvs()是基于标准正态分布产生截断的正态分布
    # 截断分布是限制x取值范围的一种分布
    # 通俗来说就是根据x范围对正态分布砍几刀挑一块出来，之后通过变换使得总面积仍然为1
    # lower = self.lbound = 0
    # upper = self.rbound = 1
    # mu = action = actor输出结果
    # scipy.stats.truncnorm.rvs()可产生服从给定分布的一个样本，通过size指定输出数组大小
    def sample_from_truncated_normal_distribution(self, lower, upper, mu, sigma, size=1):
        from scipy import stats
        return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)
