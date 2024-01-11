import random

from gymnasium.utils import seeding
from gymnasium import spaces

import numpy as np

import Discrete_Environment.AgentUtils as AgentUtils
from pettingzoo.sisl.pursuit.utils.agent_layer import AgentLayer
import pygame

import math

class Discrete:

    def __init__(self, render_mode="human", x_size=32, y_size=16, max_cycles=1000, randomSpawn=True, sparseReward=False, fullyObservable=False, competitive=False, moveTime=3, spawnTime=1, n_agents=4, obs_range=7):
        #options
        self.render_mode = render_mode
        self.x_size = x_size
        self.y_size = y_size
        self.map_matrix = np.zeros((self.x_size, self.y_size), dtype=np.int32)
        self.max_cycles = max_cycles
        self.randomSpawn = randomSpawn
        self.sparseReward = sparseReward
        self.fullyObservable = fullyObservable
        self.competitive = competitive
        self.moveTime = moveTime
        self.spawnTime = spawnTime
        self.n_agents = n_agents
        self.obs_range = obs_range

        self._seed()
        
        self.screen = None;
        
        self.obs_offset = int((self.obs_range - 1) / 2)
        
        
        
        self.availableCols = [
            (255, 0, 0),   # Red
            (0, 255, 0),   # Green
            (255, 255, 0), # Yellow
            (0, 128, 255), # Light Blue
            (255, 0, 255), # Magenta
            (255, 128, 0), # Orange
            (128, 255, 0), # Lime
            (128, 0, 128),  # Purple
            (0, 0, 255),   # Blue
            (0, 255, 255), # Cyan
        ] 
        self.currentCol = 0      
      
        
        if self.competitive:
            self.agents = AgentUtils.create_agents(
                self.n_agents, self.map_matrix, self.obs_range, self.np_random, randinit= self.randomSpawn , competitive=True
            )
        else:
            self.agents = AgentUtils.create_agents(
                self.n_agents, self.map_matrix, self.obs_range, self.np_random, randinit= self.randomSpawn 
            )
        
        if(self.competitive):
            self.availableCols = [
            (255, 0, 0),   # Red
            (0, 255, 0),   # Green
            ]
        else:
            self.availableCols = self.availableCols[:self.n_agents]
        
        

        self.agent_layer = AgentLayer(self.x_size, self.y_size, self.agents)

       
        
        n_act_agent = self.agent_layer.get_nactions(0)
        
        self.latest_reward_state = [0 for _ in range(self.n_agents)]
        self.latest_obs = [None for _ in range(self.n_agents)]
        max_agents_overlap = max(self.n_agents, self.n_agents)
        if not self.fullyObservable:
            obs_space = spaces.Box(
                low=0,
                high = max_agents_overlap,
                shape=(self.obs_range, self.obs_range, 3),
                dtype=np.float32,
            )
        else:
            obs_space = spaces.Box(
                low=0,
                high = max_agents_overlap,
                shape=(self.x_size, self.y_size, 3),
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
        
        
        self.wall_cooldown = 15
        self.wall_cooldown_counter = 0
        self.walls = []
        
        self.tinted_sprites = {}
        
        self.currentSpawnFunction = 0
        self.currentObstacleSpawnY = 0
        
        
        if(self.sparseReward):
            self.spawn_functions = [self.spawnWall, self.spawnObstacles, self.spawnObstacles]
        else:
            self.spawn_functions = [self.spawnWall, self.spawnObstacles]
        
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
        
        if self.competitive:
            self.agents = AgentUtils.create_agents(
                self.n_agents, self.map_matrix, self.obs_range, self.np_random, constraints=constraints, randinit=self.randomSpawn , competitive=True
            )
        else:
            self.agents = AgentUtils.create_agents(
                self.n_agents, self.map_matrix, self.obs_range, self.np_random, constraints=constraints, randinit=self.randomSpawn 
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
        self.fuels = []
        self.frames = 0
        self.wall_cooldown_counter = 0
        self.currentCol = 0
        self.currentObstacleSpawnY = 0

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
        
        
        if(self.sparseReward):
            for i in range(self.agent_layer.n_agents()):
                agent_layer.allies[i].fuel_deplete()

        self.model_state[1] = self.agent_layer.get_state_matrix()

        self.moveWallsStep += 1
        self.checkObstacleCollision()

        if self.moveWallsStep > self.moveTime:
            self.moveWallsStep = 0
            self.moveWalls()
            self.moveObstacles()
        self.model_state[0] = self.map_matrix
        
        self.latest_reward_state = self.reward() / self.n_agents
        self.checkWalls()
        self.checkObstacleCollision()

        if self.spawnStep >= self.spawnTime:
            self.spawnStep = 0
            if self.wall_cooldown_counter <= 0:
                if not self.randomSpawn:
                    self.spawn_functions[self.currentSpawnFunction]()
                    self.currentSpawnFunction += 1
                    if(self.currentSpawnFunction >= len(self.spawn_functions)):
                        self.currentSpawnFunction = 0
                else:
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

    def moveObstacles(self):
        new_asteroids = []
        for asteroid in self.asteriods:
            (x, y) = asteroid['position']
            new_x, new_y = x - 1, y 
            if 0 <= new_x < self.x_size:
                asteroid['position'] = (new_x, new_y)
                new_asteroids.append(asteroid)

        self.asteriods = new_asteroids
        
        new_fuel = []
        for fuel in self.fuels:
            (x, y) = fuel['position']
            new_x, new_y = x - 1, y 
            if 0 <= new_x < self.x_size:
                fuel['position'] = (new_x, new_y)
                new_fuel.append(fuel)
        
        self.fuels = new_fuel

        
        
    def checkWalls(self):
        if len(self.walls) < 1:
            return

        (wallX, wallCol) = self.walls[0]
        for i in range(self.agent_layer.n_agents()):
            agentX, agentY = self.agent_layer.get_position(i)
            agentCol = self.agent_layer.allies[i].get_color()
            if agentX > wallX and wallCol == agentCol:
                self.walls.remove((wallX, wallCol))
                break
            elif agentX > wallX:
                self.reset()
                break
            
    def checkObstacleCollision(self):
        asteroids_to_remove = []
        fuel_to_remove = []

        for i in range(self.agent_layer.n_agents()):
            agentX, agentY = self.agent_layer.get_position(i)

            for fuel in list(self.fuels):
                (fuelX, fuelY) = fuel['position']
                if agentX == fuelX and agentY == fuelY:
                    self.agent_layer.allies[i].make_fuel_full()
                    fuel_to_remove.append(fuel)

            
            for asteroid in list(self.asteriods):
                (asteroidX, asteroidY) = asteroid['position']

                if agentX == asteroidX and agentY == asteroidY:
                    self.disableAgentMovement(i, 5)
                    asteroids_to_remove.append(asteroid)

        # Remove asteroids or fuel that collided with agents
        for asteroid in asteroids_to_remove:
            if(self.asteriods.__contains__(asteroid)):
                self.asteriods.remove(asteroid)
        for fuel in fuel_to_remove:
            if(self.fuels.__contains__(fuel)):
                self.fuels.remove(fuel)

        

    def disableAgentMovement(self, agent_id, steps):
        self.agent_layer.allies[agent_id].set_disable_movement(True)
        self.agent_layer.allies[agent_id].set_disable_movement_steps(steps)
                
                
    
    def spawnWall(self):
        x_len, y_len = self.model_state[0].shape
        if(not self.randomSpawn):
            col = self.availableCols[self.currentCol]
            self.currentCol += 1
            if(self.competitive):
                if(self.currentCol >= 2):
                    self.currentCol = 0
            else:
                if(self.currentCol >= self.n_agents):
                    self.currentCol = 0
        else:
            col = random.choice(self.availableCols)
        self.walls.append((x_len - 1, col))  
        
    
    def spawnAsteroid(self, y):
        x_len, y_len = self.model_state[0].shape
        asteroid_sprite = pygame.image.load(f"Discrete_Environment/art/Asteroid{random.randint(1, 3)}.png")
        asteroid_sprite = pygame.transform.scale(
            asteroid_sprite, (int(self.pixel_scale), int(self.pixel_scale))
        )
        x = x_len - 1
        self.asteriods.append({
            'position': (x, y),
            'sprite': asteroid_sprite
        }) 
        
    def spawnFuel(self, y):
        x_len, y_len = self.model_state[0].shape
        fuel_sprite = pygame.image.load(f"Discrete_Environment/art/Fuel.png")
        fuel_sprite = pygame.transform.scale(
            fuel_sprite, (int(self.pixel_scale), int(self.pixel_scale))
        )
        x = x_len - 1
        self.fuels.append({
            'position': (x, y),
            'sprite': fuel_sprite
        })
    
    def spawnObstacles(self):
        numberOfSpawns = self.n_agents // 2
        new_asteroids = []

        for _ in range(numberOfSpawns):
            available_y_positions = set(range(self.y_size - 1))

            available_y_positions -= set(asteroid_y for (_, asteroid_y) in new_asteroids)

            if not available_y_positions:
                break

            if(not self.randomSpawn):
                y = self.currentObstacleSpawnY
                self.currentObstacleSpawnY += 1
                if(self.currentObstacleSpawnY >= self.y_size):
                    self.currentObstacleSpawnY = 0
                if(self.currentSpawnFunction == 1):
                    self.spawnAsteroid(y)
                elif(self.currentSpawnFunction == 2):
                    self.spawnFuel(y)
            else:
                y = random.choice(list(available_y_positions))
                if self.np_random.random() < 0.7:
                    self.spawnAsteroid(y)
                else:
                    self.spawnFuel(y)
            new_asteroids.append((self.x_size - 1, y))


    
    def reward(self):
        rewards = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            x, y = self.agent_layer.get_position(i)

            for (wall_x, wall_col) in self.walls:
                agent_col = self.agent_layer.allies[i].get_color()
                if x > wall_x and wall_col == agent_col:
                    rewards[i] = 20 
                    break
                elif x > wall_x and wall_col != agent_col:
                    rewards[i] = -20 
                    for j in range(self.n_agents):
                        teamnr = self.agent_layer.allies[i].get_team_nr()
                        if i != j and teamnr == self.agent_layer.allies[j].get_team_nr():
                            rewards[j] += -5
                        elif i != j:
                            rewards[j] += 20
                    break

            for asteroid in self.asteriods:
                asteroid_x, asteroid_y = asteroid['position']
                if x == asteroid_x and y == asteroid_y:
                    rewards[i] += -1 
                    break

            if rewards[i] == 0 and not self.sparseReward:
                rewards[i] = 0.01
            elif self.sparseReward:
                for fuel in self.fuels:
                    fuel_x, fuel_y = fuel['position']
                    if x == fuel_x and y == fuel_y:
                        rewards[i] +=  0.1 * self.agent_layer.allies[i].get_fuel_diff()
                        break
                
        return rewards
    
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
            
            # draw fire first
            fire_sprite_path = f"FireFull{random.choice([1, 2, 3])}.png"
            fire_sprite = pygame.image.load("Discrete_Environment/art/"+ fire_sprite_path)
            fire_sprite = pygame.transform.scale(fire_sprite, (int(self.pixel_scale), int(self.pixel_scale)))
            fire_sprite_rect = fire_sprite.get_rect(center=center)
            rotated_fire_sprite = pygame.transform.rotate(fire_sprite, rotation_angle)
            if(self.agent_layer.allies[i].get_current_fuel() > 0 or self.agent_layer.allies[i].get_disable_movement() == True):
                self.screen.blit(rotated_fire_sprite, fire_sprite_rect.topleft)


            # Rotate the tinted sprite
            rotated_sprite = pygame.transform.rotate(tinted_sprite, rotation_angle)

            # Draw the rotated sprite on the screen
            rotated_rect = rotated_sprite.get_rect(center=center)
            self.screen.blit(rotated_sprite, rotated_rect.topleft)
            
             # Display fuel slider under the agent's sprite
            if(self.sparseReward):
                fuel_percentage = self.agent_layer.allies[i].get_current_fuel() / self.agent_layer.allies[i].fullFuel
                fuel_slider_width = int(self.pixel_scale)
                fuel_slider_height = 5
                fuel_slider_pos = (center[0] - fuel_slider_width / 2, center[1] + self.pixel_scale / 2)
                pygame.draw.rect(self.screen, (0, 255, 0), (fuel_slider_pos[0], fuel_slider_pos[1], fuel_slider_width * fuel_percentage, fuel_slider_height))

                # Draw the border of the fuel slider
                pygame.draw.rect(self.screen, (255, 255, 255), (fuel_slider_pos[0], fuel_slider_pos[1], fuel_slider_width, fuel_slider_height), 1)
            
            

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
        sprite = pygame.image.load("Discrete_Environment/art/SpaceShip.png")
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
            asteroid_sprite = asteroid['sprite']

            # Calculate the position so that the center of the sprite is at (x, y)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2 - asteroid_sprite.get_width() / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2 - asteroid_sprite.get_height() / 2),
            )

            self.screen.blit(asteroid_sprite, center)
            
    def draw_fuels(self):
        for fuel in self.fuels:
            (x, y) = fuel['position']
            fuel_sprite = fuel['sprite']

            # Calculate the position so that the center of the sprite is at (x, y)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2 - fuel_sprite.get_width() / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2 - fuel_sprite.get_height() / 2),
            )

            self.screen.blit(fuel_sprite, center)
    
    def draw_grid(self):
        for x in range(self.x_size + 1):
            pygame.draw.line(
                self.screen, (255, 255, 255),
                (x * self.pixel_scale, 0),
                (x * self.pixel_scale, self.y_size * self.pixel_scale)
            )
        for y in range(self.y_size + 1):
            pygame.draw.line(
                self.screen, (255, 255, 255),
                (0, y * self.pixel_scale),
                (self.x_size * self.pixel_scale, y * self.pixel_scale)
            )
        
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
        if not self.fullyObservable:
            self.draw_agent_observations()
        self.draw_wall()
        self.draw_asteroids();
        #self.draw_grid();
        if(self.sparseReward):
            self.draw_fuels()
        
        
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
        if(self.fullyObservable):
            obs = np.zeros((3, self.x_size, self.y_size), dtype=np.float32)
        else:
            obs = np.zeros((3, self.obs_range, self.obs_range), dtype=np.float32)
        obs[0].fill(1.0) 
        xp, yp = agent_layer.get_position(agent_idx)
        if not self.fullyObservable:
            xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)
       
        if(self.fullyObservable):
            obs[0:2, :, :] = np.abs(self.model_state[0:2, :, :])
        else:
            obs[0:2, xolo:xohi, yolo:yohi] = np.abs(self.model_state[0:2, xlo:xhi, ylo:yhi])
            

        # Identify walls in the observation matrix and set their values to 3 in the third channel
        for (wall_x, wall_col) in self.walls:
            agentCol = agent_layer.allies[agent_idx].get_color()
            signifier = 1
            if(wall_col == agentCol):
                signifier = 2
                
            if(self.fullyObservable):
                obs[2, wall_x, :] = signifier
            elif xlo <= wall_x < xhi:
                obs[2, wall_x - xlo, :] = signifier
            
        for asteroid in self.asteriods:
            (asteroid_x, asteroid_y) = asteroid['position']
            if(self.fullyObservable):
                    obs[2, asteroid_x , asteroid_y ] = 3
            elif xlo <= asteroid_x < xhi and ylo <= asteroid_y < yhi:
                    obs[2, asteroid_x - xlo, asteroid_y - ylo] = 3
        
        for fuel in self.fuels:
            (fuel_x, fuel_y) = fuel['position']
            if(self.fullyObservable):
                    obs[2, fuel_x , fuel_y ] = 4
            elif xlo <= fuel_x < xhi and ylo <= fuel_y < yhi:
                    obs[2, fuel_x - xlo, fuel_y - ylo] = 4
            
        
    
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