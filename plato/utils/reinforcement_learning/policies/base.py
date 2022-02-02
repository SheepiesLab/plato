import copy
import logging
import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from plato.config import Config
from torch import nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action

        # actor NN layers, e.g.,
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.max_action * torch.tanh(self.l2(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()

        # critic NN layers, e.g.,
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = self.l2(x)
        return x


class Policy(ABC):
    def __init__(self, state_dim, action_space):
        self.max_action = Config().algorithm.max_action
        self.hidden_size = Config().algorithm.hidden_size
        self.device = Config().device()

        # Parameters for policy updating
        self.lr = Config().algorithm.learning_rate

        # initialize NNs
        self.actor = Actor(state_dim, action_space.shape[0], self.hidden_size,
                           self.max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr)

        self.critic = Critic(state_dim, action_space.shape[0],
                             self.hidden_size)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.lr)

        self.total_it = 0

        self.replay_buffer = None

    def get_initial_states(self):
        h_0, c_0 = None, None
        if Config().algorithm.recurrent_actor:
            h_0 = torch.zeros(
                (self.actor.l1.num_layers, 1, self.actor.l1.hidden_size),
                dtype=torch.float)
            # h_0 = h_0.to(self.device)

            c_0 = torch.zeros(
                (self.actor.l1.num_layers, 1, self.actor.l1.hidden_size),
                dtype=torch.float)
            # c_0 = c_0.to(self.device)
        return (h_0, c_0)

    @abstractmethod
    def select_action(self, state):

    @abstractmethod
    def update(self):

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def save_model(self, ep=None):
        """Saving the model to a file."""
        model_name = Config().algorithm.model_name
        model_path = f'./models/{model_name}/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if ep is not None:
            model_path += 'iter' + str(ep) + '_'

        torch.save(self.actor.state_dict(), model_path + 'actor.pth')
        torch.save(self.actor_optimizer.state_dict(),
                   model_path + "actor_optimizer.pth")
        torch.save(self.critic.state_dict(), model_path + 'critic.pth')
        torch.save(self.critic_optimizer.state_dict(),
                   model_path + "critic_optimizer.pth")

        logging.info("[RL Agent] Model saved to %s.", model_path)

    def load_model(self, ep=None):
        """Loading pre-trained model weights from a file."""
        model_name = Config().algorithm.model_name
        model_path = f'./models/{model_name}/'
        if ep is not None:
            model_path += 'iter' + str(ep) + '_'

        logging.info("[RL Agent] Loading a model from %s.", model_path)

        self.actor.load_state_dict(torch.load(model_path + 'actor.pth'))
        self.actor_optimizer.load_state_dict(
            torch.load(model_path + 'actor_optimizer.pth'))
        self.critic.load_state_dict(torch.load(model_path + 'critic.pth'))
        self.critic_optimizer.load_state_dict(
            torch.load(model_path + 'critic_optimizer.pth'))
