import functools
import random
from copy import copy

import gymnasium
from gymnasium.utils import seeding
from gymnasium import spaces

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.utils.env import ParallelEnv, AECEnv

from pettingzoo.sisl.pursuit.utils import agent_utils
from pettingzoo.sisl.pursuit.utils.agent_layer import AgentLayer
from pettingzoo.sisl.pursuit.utils.controllers import (
    PursuitPolicy,
    RandomPolicy,
    SingleActionPolicy,
)

import pygame
import time

class Discrete:

    def __init__(self):
            
        self.render_mode = "human"
        self.x_size = 32
        self.y_size = 16
        self.map_matrix = np.zeros((self.x_size, self.y_size), dtype=np.int32)
        self.max_cycles = 1000
        
        self._seed()
        
        
        self.n_agents = 6
        
        self.obs_range = 7
        
        self.obs_offset = int((self.obs_range - 1) / 2)
        
        self.agents = agent_utils.create_agents(
            self.n_agents, self.map_matrix, self.obs_range, self.np_random
        )
        
        self.agent_layer = AgentLayer(self.x_size, self.y_size, self.agents)
        
        
        n_act_agent = self.agent_layer.get_nactions(0)
        
        self.latest_reward_state = [0 for _ in range(self.n_agents)]
        self.latest_obs = [None for _ in range(self.n_agents)]
        max_agents_overlap = max(self.n_agents, self.n_agents)
        obs_space = spaces.Box(
            low=0,
            high = max_agents_overlap,
            shape=(self.obs_range, self.obs_range, 3),
            dtype=np.float32,
        )
        
        act_space = spaces.Discrete(n_act_agent)
        self.action_space = [act_space for _ in range(self.n_agents)]
        
        self.current_agent_layer = np.zeros((self.x_size, self.y_size), dtype=np.int32)
        
        self.agent_controller = (
                RandomPolicy(n_act_agent, self.np_random)
            )

        self.observation_space = [obs_space for _ in range(self.n_agents)]
        self.act_dims = [n_act_agent for i in range(self.n_agents)]
        
        self.model_state = np.zeros((4,) + self.map_matrix.shape, dtype=np.float32)
        self.pixel_scale = 30
        self.constraint_window = 1.0
        
    def close(self):
        pygame.event.pump()
        pygame.display.quit()
        pygame.quit()

    def reset(self):
        x_window_start = self.np_random.uniform(0.0, 1.0 - self.constraint_window)
        y_window_start = self.np_random.uniform(0.0, 1.0 - self.constraint_window)
        xlb, xub = int(self.x_size * x_window_start), int(
            self.x_size * (x_window_start + self.constraint_window)
        )
        ylb, yub = int(self.y_size * y_window_start), int(
            self.y_size * (y_window_start + self.constraint_window)
        )
        constraints = [[xlb, xub], [ylb, yub]]
        
        self.agents = agent_utils.create_agents(
            self.n_agents,
            self.map_matrix,
            self.obs_range,
            self.np_random,
            randinit=True,
            constraints=constraints,
        )
        
        self.latest_reward_state = [0 for _ in range(self.n_agents)]
        self.latest_done_state = [False for _ in range(self.n_agents)]
        self.latest_obs = [None for _ in range(self.n_agents)]
        
        self.agent_layer = AgentLayer(self.x_size, self.y_size, self.agents)

        
        self.model_state[0] = self.map_matrix
        self.model_state[1] = self.agent_layer.get_state_matrix()

        self.frames = 0

        return self.safely_observe(0)
    
    def _seed(self, seed=0):
        self.np_random, seed_ = seeding.np_random(seed)

        return [seed_]

    def step(self, action, agent_id, islast):
        
        agent_layer = self.agent_layer
        agent_layer.move_agent(agent_id, action)
        self.model_state[1] = self.agent_layer.get_state_matrix()
        
        self.model_state[0] = self.map_matrix
        
        self.render()
    
    def reward(self):
        
        return np.zeros(self.x_size, self.y_size)
    
    def draw_model_state(self):
        # -1 is building pixel flag
        x_len, y_len = self.model_state[0].shape
        for x in range(x_len):
            for y in range(y_len):
                pos = pygame.Rect(
                    self.pixel_scale * x,
                    self.pixel_scale * y,
                    self.pixel_scale,
                    self.pixel_scale,
                )
                col = (255, 255, 255)
                if self.model_state[0][x][y] == -1:
                    col = (255, 255, 255)
                pygame.draw.rect(self.screen, col, pos)
    
    def draw_agents(self):
        for i in range(self.agent_layer.n_agents()):
            x, y = self.agent_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            col = (255, 0, 0)
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))

    def render(self):
        
        if self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
            )
        else:
            self.screen = pygame.Surface(
                (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
            )
        
        
        #self.draw_model_state()
        self.draw_agents()
        
        
        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if self.render_mode == "human":
            pygame.display.flip()
        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def safely_observe(self, i):
        agent_layer = self.agent_layer
        obs = self.collect_obs(agent_layer, i)
        return obs
    
    def collect_obs(self, agent_layer, i):
        for j in range(self.n_agents):
            if i == j:
                return self.collect_obs_by_idx(agent_layer, i)
        assert False, "bad index"
        
    def collect_obs_by_idx(self, agent_layer, agent_idx):
        # returns a flattened array of all the observations
        obs = np.zeros((3, self.obs_range, self.obs_range), dtype=np.float32)
        obs[0].fill(1.0)  # border walls set to -0.1?
        xp, yp = agent_layer.get_position(agent_idx)

        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)

        obs[0:3, xolo:xohi, yolo:yohi] = np.abs(self.model_state[0:3, xlo:xhi, ylo:yhi])
        return obs
    
    def obs_clip(self, x, y):
        xld = x - self.obs_offset
        xhd = x + self.obs_offset
        yld = y - self.obs_offset
        yhd = y + self.obs_offset
        xlo, xhi, ylo, yhi = (
            np.clip(xld, 0, self.x_size - 1),
            np.clip(xhd, 0, self.x_size - 1),
            np.clip(yld, 0, self.y_size - 1),
            np.clip(yhd, 0, self.y_size - 1),
        )
        xolo, yolo = abs(np.clip(xld, -self.obs_offset, 0)), abs(
            np.clip(yld, -self.obs_offset, 0)
        )
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1