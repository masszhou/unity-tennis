# author: Zhiliang Zhou, zhouzhiliang@gmail.com
# learned from Udacity DRLND showcase codes

import numpy as np
import random
from collections import deque, namedtuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchsummary as TorchSummary

from utilities import hard_update, soft_update
from networks import Actor
from networks import Critic
# add OU noise for exploration
from noise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG:
    def __init__(self,
                 in_actor,
                 out_actor,
                 in_critic,  # e.g. = n_agent * (state_size + action_size)
                 gamma=0.99,  # discount factor
                 tau=1e-3,  # for soft update of target network parameters
                 lr_actor=1e-4,
                 lr_critic=1e-3,  # better learn faster than actor
                 random_seed=2):
        self.state_size = in_actor
        self.action_size = out_actor
        self.seed = random.seed(random_seed)

        self.params = {"lr_actor": lr_actor,
                       "lr_critic": lr_critic,
                       "gamma": gamma,
                       "tau": tau,
                       "optimizer": "adam"}

        self.local_actor = Actor(in_shape=in_actor, out_shape=out_actor).to(device)
        self.target_actor = Actor(in_shape=in_actor, out_shape=out_actor).to(device)
        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=lr_actor)

        # for a single agent, critic takes global observations as input, and output action-value Q
        # e.g. global_states = all_states + all_actions
        self.local_critic = Critic(in_shape=in_critic).to(device)
        self.target_critic = Critic(in_shape=in_critic).to(device)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=lr_critic)

        # Q: should local/target start with same weights ? synchronized after first copy after all
        # A: better hard copy at the beginning
        hard_update(self.target_actor, self.local_actor)
        hard_update(self.target_critic, self.local_critic)

        # Noise process
        self.noise = OUNoise(out_actor, scale=1.0)

    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.local_actor(obs) + noise * self.noise.noise()
        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise * self.noise.noise()
        return action


class MADDPG:
    def __init__(self,
                 state_size=24,
                 action_size=2,
                 n_agents=2,
                 memory_size=int(1e5),  # replay buffer size
                 batch_size=128):       # minibatch size

        self.batch_size = batch_size
        self.param = dict(state_size=state_size,
                          action_size=action_size,
                          n_agents=n_agents,
                          memory_size=memory_size,
                          batch_size=batch_size)

        agent_config = dict(in_actor=state_size,
                            out_actor=action_size,
                            in_critic=n_agents*(state_size+action_size),
                            gamma=0.99,
                            tau=0.01,
                            lr_actor=0.0001,
                            lr_critic=0.0001)
        self.agent_pool = [DDPG(**agent_config) for i in range(n_agents)]  # list of DDPG agent

        self.memory = ReplayBuffer(memory_size, batch_size)

        self.local_step = 0
        self.target_step = 0

    def reset(self):
        #self.noise.reset()
        pass

    def act(self, state, add_noise=True):
        # # for single agent only
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # self.actor_local.eval()  # must set to eval mode, since BatchNorm used
        # with torch.no_grad():
        #     action = self.actor_local(state).cpu().data.numpy()
        # self.actor_local.train()
        # if add_noise:
        #     action += self.noise.sample()
        # return np.clip(action.squeeze(), -1, 1)
        pass

    def eval_local_act(self, obs_all_agents, noise=0.0):
        """
        get actions from all agents in the MADDPG object

        :param obs_all_agents: tensor, rank=3, shape=[batch_size, n_agents, in_critic]
        :param noise:
        :return:
        """
        actions = [ddpg_agent.act(obs, noise) for ddpg_agent, obs in zip(self.agent_pool, obs_all_agents.transpose(0, 1))]
        return actions

    def eval_target_act(self, obs_all_agents, noise=0.0):
        """
        get target network actions from all the agents in the MADDPG object

        :param obs_all_agents: tensor, rank=3, shape=[batch_size, n_agents, in_critic]
        :param noise:
        :return: list of tensor, tensor rank=2, shape=[batch_size, out_actor]
        """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.agent_pool, obs_all_agents.transpose(0, 1))]
        return target_actions

    def step(self, state, action, reward, next_state, done):
        # self.memory.add(state, action, reward, next_state, done)
        #
        # if len(self.memory) > self.params["batch_size"]:
        #     experiences = self.memory.sample()  # list of tensors
        #     self.learn(experiences, self.params["gamma"])
        pass

    def update_agent_with_id(self, experiences, agent_id, gamma, logger=None):
        """
        :param experiences:
            type = list of torch tensors
            [states, actions, rewards, next_states, dones]
            states  -> rank=3, shape=[batch_size, agent_id, state_size]
            actions -> rank=3, shape=[batch_size, agent_id, action_size]
            rewards -> rank=2, shape=[batch_size, agent_id]
            next_states -> rank=3, shape=[batch_size, agent_id, state_size]
            dones   -> rank=2, shape=[batch_size, agent_id]
        :param gamma: reward discounter
        :return:
        """
        # use this experience
        # states  -> rank=3, shape=[batch_size, agent_id, state_size]
        # actions -> rank=3, shape=[batch_size, agent_id, action_size]
        # rewards -> rank = 2, shape = [batch_size, agent_id]
        # next_states -> rank = 3, shape = [batch_size, agent_id, state_size]
        # dones   -> rank=2, shape=[batch_size, agent_id]
        states, actions, rewards, next_states, dones = experiences

        # update this agent
        agent = self.agent_pool[agent_id]

        # ------------------------------------------
        # update centralized critic
        # ------------------------------------------
        # -- recall update critic for normal DDPG
        # best_actions = self.actor_target(next_states)
        # Q_next_max = self.critic_target(next_states, best_actions)
        # Q_target = rewards + gamma * Q_next_max * (1 - dones)
        # Q_local = self.critic_local(states, actions)
        # critic_loss = F.mse_loss(Q_local, Q_target.detach())
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()

        target_actions = self.eval_target_act(next_states)  # list of tensor, tensor rank=2, shape=[batch_size, out_actor]
        target_actions = torch.cat(target_actions, dim=1)  # tensor, rank=2, shape=[batch_size, n_agent*action_size]

        target_critic_input = torch.cat((next_states.view(self.batch_size, -1),  # tensor, rank=2, shape=[batch_size, n_agent*state_size]
                                         target_actions),  # tensor, rank=2, shape=[batch_size, n_agent*action_size]
                                        dim=1).to(device)
        # tensor, rank=2, shape=[batch_size, n_agent*(state_size+action_size)]

        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)  # rank=2, shape=[batch_size, 1]

        agent_rewards = rewards[:, agent_id]  # rank = 1, shape = [batch_size]
        agent_dones = dones[:, agent_id]  # rank = 1, shape = [batch_size]

        q_target = agent_rewards.view(-1, 1) + gamma * q_next * (1 - agent_dones.view(-1, 1))  # rank=2, shape=[batch_size, 1]

        local_critic_input = torch.cat((states.view(self.batch_size, -1),  # tensor, rank=2, shape=[batch_size, n_agent*state_size]
                                        actions.view(self.batch_size, -1)),  # tensor, rank=2, shape=[batch_size, n_agent*action_size]
                                       dim=1).to(device)
        q_local = agent.local_critic(local_critic_input)

        critic_loss = F.mse_loss(q_local, q_target.detach())

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        # ------------------------------------------
        # update actor
        # ------------------------------------------
        # -- recall update actor for normal DDPG
        # actions_pred = self.actor_local(states)
        # Q_baseline = self.critic_local(states, actions_pred)
        # actor_loss = -Q_baseline.mean()  # I think this is a good trick to make loss to scalar
        # # note, gradients from both actor_local and critic_local will be calculated
        # # however we only update actor_local
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()

        pred_actions = self.eval_local_act(states)   # list of tensor, tensor rank=2, shape=[batch_size, out_actor]
        pred_actions = torch.cat(pred_actions, dim=1)  # tensor, rank=2, shape=[batch_size, n_agent*action_size]

        local_critic_input2 = torch.cat((states.view(self.batch_size, -1),  # tensor, rank=2, shape=[batch_size, n_agent*state_size]
                                         pred_actions),  # tensor, rank=2, shape=[batch_size, n_agent*action_size]
                                        dim=1).to(device)

        q_baseline = agent.local_critic(local_critic_input2)

        # get the policy gradient
        actor_loss = -q_baseline.mean()  # scalar trick for gradients
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        self.local_step += 1

        # tensorboardX addon
        if logger is not None:
            al = actor_loss.cpu().detach().item()
            cl = critic_loss.cpu().detach().item()
            logger.add_scalars('agent%i/losses' % agent_id,
                               {'critic loss': cl,
                                'actor_loss': al},
                               self.local_step)

    def update_targets(self):
        """
        soft update targets
        note, in the showcase code of openai, suggest update target network every n-episodes
        """
        self.target_step += 1
        for ddpg_agent in self.agent_pool:
            soft_update(ddpg_agent.target_actor, ddpg_agent.local_actor, self.param["tau"])
            soft_update(ddpg_agent.target_critic, ddpg_agent.local_critic, self.param["tau"])


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed=42):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """

        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # stacked with new axis
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        # stacked with new axis
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None])).float().to(device)
        # vstacked with old axis
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # stacked with new axis
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        # vstacked with old axis
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


if __name__ == "__main__":
    pass