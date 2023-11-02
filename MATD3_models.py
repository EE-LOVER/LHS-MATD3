# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 23:13:31 2022

@author: 38688
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, hidden_width)
        self.l4 = nn.Linear(hidden_width, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.leaky_relu(self.l1(s))
        s = F.leaky_relu(self.l2(s))
        s = F.leaky_relu(self.l3(s))
        s = F.leaky_relu(self.l4(s))
        s = F.leaky_relu(self.l5(s))
        a = (torch.tanh(self.l6(s))+1)/2
        return a


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, hidden_width)
        self.l4 = nn.Linear(hidden_width, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)
        # Q2
        self.l7 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l8 = nn.Linear(hidden_width, hidden_width)
        self.l9 = nn.Linear(hidden_width, hidden_width)
        self.l10 = nn.Linear(hidden_width, hidden_width)
        self.l11 = nn.Linear(hidden_width, hidden_width)
        self.l12 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.leaky_relu(self.l1(s_a))
        q1 = F.leaky_relu(self.l2(q1))
        q1 = F.leaky_relu(self.l3(q1))
        q1 = F.leaky_relu(self.l4(q1))
        q1 = F.leaky_relu(self.l5(q1))
        q1 = self.l6(q1)

        q2 = F.leaky_relu(self.l7(s_a))
        q2 = F.leaky_relu(self.l8(q2))
        q2 = F.leaky_relu(self.l9(q2))
        q2 = F.leaky_relu(self.l10(q2))
        q2 = F.leaky_relu(self.l11(q2))
        q2 = self.l12(q2)

        return q1, q2

    def Q1(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.leaky_relu(self.l1(s_a))
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

    def store(self, s, a, r, s_, s2, s2_):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.s2[self.count] = s2
        self.s2_[self.count] = s2_
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

        return batch_s, batch_a, batch_r, batch_s_, batch_s2, batch_s2_


class MATD3(object):
    def __init__(self, case):
        self.agent_num = len(case.strategy_id)
        self.hidden_width_actor = 128  # The number of neurons in hidden layers of the neural network
        self.hidden_width_critic = 128
        self.memory_size = 128 
        self.batch_size = 64  # batch size
        self.GAMMA = 0.3 # discount factor
        self.TAU = 0.05  # Softly update the target network
        self.lr = 5e-5 # learning rate
        self.policy_noise = 0.0001  # The noise for the trick 'target policy smoothing'
        self.noise_clip = 0.05  # Clip the noise
        self.policy_freq = 2  # The frequency of policy updates
        self.actor_pointer = 0
        
        self.actor_n = []
        self.actor_target_n = []
        self.critic_n = []
        self.critic_target_n = []
        self.actor_optimizer_n = []
        self.critic_optimizer_n = []
        self.state_dim_G = case.state_dim_G
        self.action_dim_G = case.action_dim_G
        self.state_dim2 = case.state_dim2
        self.all_action_dim = self.action_dim_G*self.agent_num
        for i in range(self.agent_num):
            actor = Actor(self.state_dim_G, self.action_dim_G, self.hidden_width_actor)
            actor_target = copy.deepcopy(actor)
            critic = Critic(self.state_dim2, self.all_action_dim, self.hidden_width_critic)
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
    
    def choose_action_random_G(self):
        a = np.random.rand(self.agent_num,self.action_dim_G)
        return a

    def choose_action_G(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = np.zeros((self.agent_num,self.action_dim_G))
        for i in range(self.agent_num):
            a[i,:] = self.actor_n[i](s).data.numpy().flatten()
        return a
    
    def choose_action_with_noise_G(self, s, noise):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = np.zeros((self.agent_num,self.action_dim_G))
        for i in range(self.agent_num):
            a[i,:] = (self.actor_n[i](s).data.numpy().flatten() + 
                      np.random.normal(0, noise, size=self.action_dim_G)).clip(0, 1)
        return a

    def learn(self):
        self.actor_pointer = (self.actor_pointer + 1)%self.policy_freq
        batch_s, batch_a, batch_r, batch_s_, batch_s2, batch_s2_ = \
            self.replay_buffer.sample(self.batch_size)  # Sample a batch
        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            # Trick 1:target policy smoothing
            # torch.randn_like can generate random numbers sampled from N(0,1)，which have the same size as 'batch_a'
            noise = (torch.randn_like(batch_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = torch.tensor([])
            for i in range(self.agent_num):
                next_action = torch.cat((next_action,self.actor_target_n[i](batch_s_)),1)
            next_action = (next_action + noise).clamp(0,1)

            # Trick 2:clipped double Q-learning
            target_Q1_n = []
            target_Q2_n = []
            target_Q_n = []
            for i in range(self.agent_num):
                target_Q1, target_Q2 = self.critic_target_n[i](batch_s2_, next_action)
                target_Q = batch_r[:,i].reshape((-1,1)) + self.GAMMA * torch.min(target_Q1, target_Q2)
                target_Q1_n.append(target_Q1)
                target_Q2_n.append(target_Q2)
                target_Q_n.append(target_Q)

        # Get the current Q
        for i in range(self.agent_num):
            current_Q1, current_Q2 = self.critic_n[i](batch_s2, batch_a)
            # Compute the critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q_n[i]) + \
                F.mse_loss(current_Q2, target_Q_n[i])
            print(critic_loss)
            # Optimize the critic
            self.critic_optimizer_n[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizer_n[i].step()

        if self.actor_pointer == 0:        
            action_index = 0
            # Trick 3:delayed policy updates
            for i in range(self.agent_num):
                action_index_next = action_index + self.action_dim_G
                batch_a_copy = 1.0*batch_a
                action_agent = self.actor_n[i](batch_s_)
                batch_a_copy[:,action_index:action_index_next] = action_agent
                action_index = action_index_next
            # Freeze critic networks so you don't waste computational effort
                for params in self.critic_n[i].parameters():
                    params.requires_grad = False
                # Compute actor loss
                actor_loss = -self.critic_n[i].Q1(batch_s2, batch_a_copy).mean()  # Only use Q1
                # Optimize the actor
                self.actor_optimizer_n[i].zero_grad()
                actor_loss.backward()
                self.actor_optimizer_n[i].step()
    
                # Unfreeze critic networks
                for params in self.critic_n[i].parameters():
                    params.requires_grad = True
    
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