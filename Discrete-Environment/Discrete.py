import functools
import random
from copy import copy

import gymnasium
from gymnasium.utils import seeding
from gymnasium import spaces

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.utils.env import ParallelEnv, AECEnv

#from pettingzoo.sisl.pursuit.utils import agent_utils
import AgentUtils
from pettingzoo.sisl.pursuit.utils.agent_layer import AgentLayer
from pettingzoo.sisl.pursuit.utils.controllers import (
    PursuitPolicy,
    RandomPolicy,
    SingleActionPolicy,
)

import pygame

from pygame import Surface
from pygame.surfarray import pixels_alpha

import time
import math

class Discrete:

    def __init__(self):
            
        self.render_mode = "human"
        self.x_size = 32
        self.y_size = 16
        self.map_matrix = np.zeros((self.x_size, self.y_size), dtype=np.int32)
        self.max_cycles = 1000
        
        self._seed()
        
        self.screen = None;
        self.n_agents = 6
        
        self.obs_range = 7
        
        self.obs_offset = int((self.obs_range - 1) / 2)
        
        self.availableCols = [(255,0,0), (0,255,0), (0,0,255),(255,255,0), (0,255,255), (255,0,255),(128,0,0), (0,128,0), (0,0,128), (128,128,0)]
        
        self.agents = AgentUtils.create_agents(
            self.n_agents, self.map_matrix, self.obs_range, self.np_random, self.availableCols
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
        
        #self.agent_controller = (
        #        RandomPolicy(n_act_agent, self.np_random)
         #   )

        self.observation_space = [obs_space for _ in range(self.n_agents)]
        self.act_dims = [n_act_agent for i in range(self.n_agents)]
        
        self.model_state = np.zeros((4,) + self.map_matrix.shape, dtype=np.float32)
        self.pixel_scale = 30
        self.constraint_window = 1.0
        self.spawnStep = 0
        self.moveWallsStep = 0
        
        
        self.wall_cooldown = 30
        self.wall_cooldown_counter = 0
        self.walls = []
        
        self.tinted_sprites = {}
        
    def close(self):
        pygame.event.pump()
        pygame.display.quit()
        pygame.quit()

    def reset(self):
        self.screen = None;

        x_window_start = self.np_random.uniform(0.0, 1.0 - self.constraint_window)
        y_window_start = self.np_random.uniform(0.0, 1.0 - self.constraint_window)
        xlb, xub = int(self.x_size * x_window_start), int(
            self.x_size * (x_window_start + self.constraint_window)
        )
        ylb, yub = int(self.y_size * y_window_start), int(
            self.y_size * (y_window_start + self.constraint_window)
        )
        constraints = [[xlb, xub], [ylb, yub]]
        
        self.agents = AgentUtils.create_agents(
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
        self.spawnStep = 0
        self.moveWallsStep = 0
        
        self.model_state[0] = self.map_matrix
        self.model_state[1] = self.agent_layer.get_state_matrix()

        self.walls = []
        self.asteriods = []
        self.frames = 0
        self.wall_cooldown_counter = 0
        print("Reset")

        return self.safely_observe(0)
    
    def draw_agent_observations(self):
        for i in range(self.agent_layer.n_agents()):
            x, y = self.agent_layer.get_position(i)
            agent_color = self.agent_layer.allies[i].get_color()

            patch = pygame.Surface(
                (self.pixel_scale * self.obs_range, self.pixel_scale * self.obs_range)
            )
            patch.set_alpha(75)
            patch.fill(agent_color)

            ofst = self.obs_range / 2.0
            self.screen.blit(
                patch,
                (
                    self.pixel_scale * (x - ofst + 1 / 2),
                    self.pixel_scale * (y - ofst + 1 / 2),
                ),
            )
    
    def _seed(self, seed=0):
        self.np_random, seed_ = seeding.np_random(seed)

        return [seed_]

    def step(self, action, agent_id, islast):
        agent_layer = self.agent_layer
        agent_layer.move_agent(agent_id, action)
        self.model_state[1] = self.agent_layer.get_state_matrix()

        self.moveWallsStep += 1
        print("move step ", self.moveWallsStep)

        if self.moveWallsStep > 3:
            print("move step")
            self.moveWallsStep = 0
            self.moveWalls()
            self.moveAsteroids()
        self.model_state[0] = self.map_matrix
        
        self.checkWalls()


        if self.spawnStep % 5 == 0:
            if self.wall_cooldown_counter <= 0:

                if self.np_random.random() < 0.5:
                    self.spawnWall()
                else:
                    self.spawnObstacles()

                self.wall_cooldown_counter = self.wall_cooldown
            else:
                self.wall_cooldown_counter -= 1


        self.spawnStep += 1
        self.render()
    
    def moveWalls(self):
        for i in range(len(self.walls)):
            (x, col) = self.walls[i]
            self.walls[i] = (x - 1, col)
            #print(x + 1)

    def moveAsteroids(self):
        for i in range(len(self.asteriods)):
            asteroid = self.asteriods[i]
            (x, y) = asteroid['position']
            new_x, new_y = x - 1, y  # Update the position based on your desired logic
            asteroid['position'] = (new_x, new_y)
        
    def checkWalls(self):
        if len(self.walls) < 1:
            return

        (wallX, wallCol) = self.walls[0]
        for i in range(self.agent_layer.n_agents()):
            agentX, agentY = self.agent_layer.get_position(i)
            agentCol = self.agent_layer.allies[i].get_color()
            if agentX > wallX and wallCol == agentCol:
                print("RightColorWall")
                self.walls.remove((wallX, wallCol))
                break
            elif agentX > wallX:
                print("WrongColorWall")
                self.reset()
                break
                
                
    
    def spawnWall(self):
        x_len, y_len = self.model_state[0].shape
        col = random.choice(self.availableCols)
        self.walls.append((x_len, col))  
        
    
    def spawnAsteroid(self, y):
        x_len, y_len = self.model_state[0].shape
        asteroid_sprite = pygame.image.load(f"art/Asteroid{random.randint(1, 3)}.png")
        asteroid_sprite = pygame.transform.scale(
            asteroid_sprite, (int(self.pixel_scale), int(self.pixel_scale))
        )
        self.asteriods.append({
            'position': (x_len, y),
            'sprite': asteroid_sprite
        }) 
    
    def spawnObstacles(self):
        numberOfSpawns = self.n_agents // 2
        new_asteroids = []

        for _ in range(numberOfSpawns):
            available_y_positions = set(range(self.y_size - 1))

            available_y_positions -= set(asteroid_y for (_, asteroid_y) in new_asteroids)

            if not available_y_positions:
                break

            y = random.choice(list(available_y_positions))
            self.spawnAsteroid(y)
            new_asteroids.append((self.x_size, y))


    
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

            # Retrieve or create the tinted sprite for the agent's color
            agent_color = self.agent_layer.allies[i].get_color()
            if agent_color not in self.tinted_sprites:
                self.tinted_sprites[agent_color] = self.create_tinted_sprite(agent_color)

            tinted_sprite = self.tinted_sprites[agent_color]

            # Calculate the rotation angle based on the agent's movement direction
            rotation_angle = self.calculate_rotation_angle(i)

            # Rotate the tinted sprite
            rotated_sprite = pygame.transform.rotate(tinted_sprite, rotation_angle)

            # Draw the rotated sprite on the screen
            rotated_rect = rotated_sprite.get_rect(center=center)
            self.screen.blit(rotated_sprite, rotated_rect.topleft)

    def calculate_rotation_angle(self, agent_idx):
        # Get the agent's last position from the DiscreteAgent instance
        last_x, last_y = self.agent_layer.allies[agent_idx].last_position()

        # Calculate the rotation angle based on the agent's movement direction
        x, y = self.agent_layer.get_position(agent_idx)
        dx, dy = x - last_x, y - last_y

        # Calculate the angle in degrees
        angle = math.degrees(math.atan2(dy, dx))

        # Pygame uses a different coordinate system for angles (clockwise starting from the right),
        # so we need to adjust the angle accordingly
        angle = -angle + 270

        return angle

    def create_tinted_sprite(self, color):
        # Load the sprite image and adjust its size
        sprite = pygame.image.load("art/SpaceShip.png")
        sprite = pygame.transform.scale(sprite, (int(self.pixel_scale), int(self.pixel_scale)))

        # Create a copy of the sprite to avoid modifying the original
        tinted_sprite = sprite.copy()

        # Iterate through each pixel and multiply its RGB values with the agent's color
        for y in range(tinted_sprite.get_height()):
            for x in range(tinted_sprite.get_width()):
                r, g, b, a = tinted_sprite.get_at((x, y))
                tinted_sprite.set_at((x, y), (r * color[0] // 255, g * color[1] // 255, b * color[2] // 255, a))

        return tinted_sprite
            
    def draw_wall(self):
        for i in range(len(self.walls)):      
            x_len, y_len = self.model_state[0].shape
            (x,col) = self.walls[i]
            center1 = (
                    int(self.pixel_scale * x + self.pixel_scale / 2),
                    int(self.pixel_scale * 0),
                )
            center2 = (
                    int(self.pixel_scale * x + self.pixel_scale / 2),
                    int(self.pixel_scale * y_len),
                )
            pygame.draw.line(self.screen, col, center1, center2, int(self.pixel_scale / 2) )
            
    def draw_asteroids(self):
        for asteroid in self.asteriods:
            (x, y) = asteroid['position']
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            asteroid_sprite = asteroid['sprite']
            self.screen.blit(asteroid_sprite, center)
        
    def render(self):
        
        if self.screen is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
                )
                pygame.display.set_caption("Pursuit")
            else:
                self.screen = pygame.Surface(
                    (self.pixel_scale * self.x_size, self.pixel_scale * self.y_size)
                )
        
        self.screen.fill((0, 0, 0))
        #self.draw_model_state()
        self.draw_agents()
        self.draw_agent_observations()
        self.draw_wall()
        self.draw_asteroids();
        
        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
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