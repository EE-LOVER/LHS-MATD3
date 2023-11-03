# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:11:20 2022

@author: 38688
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions import Beta


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, hidden_width)
        self.l4 = nn.Linear(hidden_width, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6_alpha = nn.Linear(hidden_width, action_dim)
        self.l6_beta = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.leaky_relu(self.l1(s))
        s = F.leaky_relu(self.l2(s))
        s = F.leaky_relu(self.l3(s))
        s = F.leaky_relu(self.l4(s))
        s = F.leaky_relu(self.l5(s))
        alpha = F.softplus(self.l6_alpha(s)) + 1.0
        beta = F.softplus(self.l6_beta(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_width):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, hidden_width)
        self.l4 = nn.Linear(hidden_width, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)

    def forward(self, s):
        q1 = F.leaky_relu(self.l1(s))
        q1 = F.leaky_relu(self.l2(q1))
        q1 = F.leaky_relu(self.l3(q1))
        q1 = F.leaky_relu(self.l4(q1))
        q1 = F.leaky_relu(self.l5(q1))
        q1 = self.l6(q1)
        return q1


class ReplayBuffer(object):
    def __init__(self, max_size, state_dim, action_dim, agent_num, state_dim2):
        self.max_size = int(max_size)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, agent_num))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.s2 = np.zeros((self.max_size, state_dim2))
        self.s2_ = np.zeros((self.max_size, state_dim2))
        self.a_logprob = np.zeros((self.max_size, action_dim))

    def store(self, s, a, r, s_, s2, s2_, a_logprob):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.s2[self.count] = s2
        self.s2_[self.count] = s2_
        self.a_logprob[self.count] = a_logprob
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_s2 = torch.tensor(self.s2[index], dtype=torch.float)
        batch_s2_ = torch.tensor(self.s2_[index], dtype=torch.float)
        batch_a_logprob = torch.tensor(self.a_logprob[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_s2, batch_s2_, batch_a_logprob


class MAPPO(object):
    def __init__(self, case):
        self.agent_num = len(case.strategy_id)
        self.hidden_width_actor = 128  # The number of neurons in hidden layers of the neural network
        self.hidden_width_critic = 128
        self.memory_size = 128  # batch size
        self.batch_size = 64  # batch size
        self.GAMMA = 0.3  # discount factor
        self.TAU = 1  # Softly update the target network
        self.lr = 5e-5  # learning rate
        self.actor_pointer = 0
        self.entropy_coef = 0.01
        self.epsilon = 0.2

        self.actor_n = []
        self.actor_target_n = []
        self.critic_n = []
        self.critic_target_n = []
        self.actor_optimizer_n = []
        self.critic_optimizer_n = []
        self.state_dim_G = case.state_dim_G
        self.action_dim_G = case.action_dim_G
        self.state_dim2 = case.state_dim2
        self.all_action_dim = self.action_dim_G * self.agent_num
        for i in range(self.agent_num):
            actor = Actor(self.state_dim_G, self.action_dim_G, self.hidden_width_actor)
            actor_target = copy.deepcopy(actor)
            critic = Critic(self.state_dim2, self.hidden_width_critic)
            critic_target = copy.deepcopy(critic)
            self.actor_n.append(actor)
            self.actor_target_n.append(actor_target)
            self.critic_n.append(critic)
            self.critic_target_n.append(critic_target)
            self.actor_optimizer_n.append(torch.optim.Adam(self.actor_n[i].parameters(), lr=self.lr))
            self.critic_optimizer_n.append(torch.optim.Adam(self.critic_n[i].parameters(), lr=self.lr))
        self.replay_buffer = ReplayBuffer(self.memory_size, self.state_dim_G,
                                          self.all_action_dim, self.agent_num,
                                          self.state_dim2)

    def choose_action_random_G(self,s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = np.random.rand(self.agent_num, self.action_dim_G)
        a_logprob = np.zeros((self.agent_num, self.action_dim_G))
        for i in range(self.agent_num):
            dist = self.actor_n[i].get_dist(s)
            a_logprob[i, :] = dist.log_prob(torch.tensor(a[i, :],dtype=torch.float)).data.numpy().flatten()
        return a, a_logprob

    def choose_action_G(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = np.zeros((self.agent_num, self.action_dim_G))
        a_logprob = np.zeros((self.agent_num, self.action_dim_G))
        for i in range(self.agent_num):
            dist = self.actor_n[i].get_dist(s)
            a[i, :] = dist.sample().data.numpy().flatten()
            a_logprob[i, :] = dist.log_prob(torch.tensor(a[i, :],dtype=torch.float)).data.numpy().flatten()
        return a, a_logprob

    def choose_action_with_noise_G(self, s, noise):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = np.zeros((self.agent_num, self.action_dim_G))
        a_logprob = np.zeros((self.agent_num, self.action_dim_G))
        for i in range(self.agent_num):
            dist = self.actor_n[i].get_dist(s)
            a[i, :] = dist.sample().data.numpy().flatten()
            a_logprob[i, :] = dist.log_prob(torch.tensor(a[i, :],dtype=torch.float)).data.numpy().flatten()
        return a, a_logprob

    def learn(self):
        batch_s, batch_a, batch_r, batch_s_, batch_s2, batch_s2_, batch_a_logprob = \
            self.replay_buffer.sample(self.batch_size)  # Sample a batch
        action_index = 0

        for i in range(self.agent_num):

            adv = []
            gae = 0
            with torch.no_grad():  # adv and v_target have no gradient
                vs = self.critic_n[i](batch_s)
                vs_ = self.critic_n[i](batch_s_)
                deltas = batch_r[:, i].reshape((-1, 1)) + self.GAMMA * vs_ - vs
                for delta in reversed(deltas.flatten().numpy()):
                    gae = delta + self.GAMMA * 0.95 * gae
                    adv.insert(0, gae)
                adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
                v_target = adv + vs

            action_index_next = action_index + self.action_dim_G

            dist_now = self.actor_n[i].get_dist(batch_s)
            dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
            a_logprob_now = dist_now.log_prob(batch_a[:, action_index:action_index + self.action_dim_G])
            ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - batch_a_logprob.sum(1, keepdim=True))

            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv
            actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
            action_index = action_index_next
            # Freeze critic networks so you don't waste computational effort
            for params in self.critic_n[i].parameters():
                params.requires_grad = False
            # Optimize the actor
            self.actor_optimizer_n[i].zero_grad()
            actor_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.actor_n[i].parameters(), 0.5)
            self.actor_optimizer_n[i].step()
            # Unfreeze critic networks
            for params in self.critic_n[i].parameters():
                params.requires_grad = True

            v_s = self.critic_n[i](batch_s)
            critic_loss = F.mse_loss(v_target, v_s)
            # Update critic
            self.critic_optimizer_n[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_n[i].parameters(), 0.5)
            self.critic_optimizer_n[i].step()

            # Softly update the target networks
            for param, target_param in zip(self.critic_n[i].parameters(),
                                           self.critic_target_n[i].parameters()):
                target_param.data.copy_(self.TAU * param.data +
                                        (1 - self.TAU) * target_param.data)

            for param, target_param in zip(self.actor_n[i].parameters(),
                                           self.actor_target_n[i].parameters()):
                target_param.data.copy_(self.TAU * param.data +
                                        (1 - self.TAU) * target_param.data)

    def lr_decay(self):
        for i in range(self.agent_num):
            for p in self.actor_optimizer_n[i].param_groups:
                p['lr'] *= 0.9
            for p in self.critic_optimizer_n[i].param_groups:
                p['lr'] *= 0.9