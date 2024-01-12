import numpy as np
from gymnasium import spaces

from pettingzoo.sisl._utils import Agent

class DiscreteAgent(Agent):
    # constructor
    def __init__(
        self,
        xs,
        ys,
        map_matrix,
        randomizer,
        obs_range=3,
        n_channels=3,
        seed=1,
        flatten=False,
        col=(0, 0, 0),
        team_nr=0,
        all_agents=None 
    ):
        self.all_agents = all_agents  
        self.color = col

        self.random_state = randomizer
        
        self.disable_movement = False
        self.disable_movement_steps = 0

        self.xs = xs
        self.ys = ys

        self.eactions = [
            0,  # move left
            1,  # move right
            2,  # move up
            3,  # move down
            4, # stay
        ]  

        self.motion_range = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]]

        self.current_pos = np.zeros(2, dtype=np.int32)  # x and y position
        self.last_pos = np.zeros(2, dtype=np.int32)
        self.temp_pos = np.zeros(2, dtype=np.int32)

        self.map_matrix = map_matrix

        self.terminal = False

        self._obs_range = obs_range
        
        self.fullFuel = 150
        self.currentFuel = self.fullFuel
        
        self.team_nr = team_nr

        if flatten:
            self._obs_shape = (n_channels * obs_range**2 + 1,)
        else:
            self._obs_shape = (obs_range, obs_range, 4)

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=self._obs_shape)

    @property
    def action_space(self):
        return spaces.Discrete(5)

    # Dynamics Functions
    def step(self, a):
        cpos = self.current_pos
        lpos = self.last_pos
        # if dead or out of fuel dont move
        if self.terminal or self.currentFuel <= 0:
            return cpos
        if self.disable_movement_steps > 0:
            self.disable_movement_steps -= 1
            self.disable_movement = True
            return cpos
        elif self.disable_movement == True:
            self.disable_movement = False
            
        tpos = self.temp_pos
        tpos[0] = cpos[0]
        tpos[1] = cpos[1]

        # transition is deterministic
        tpos += self.motion_range[a]
        x = tpos[0]
        y = tpos[1]

        # check bounds
        if not self.inbounds(x, y):
            return cpos
        
        # check if the position is occupied by another agent
        for agent in self.all_agents:
            if np.array_equal(agent.current_position(), tpos):
                return cpos  
        else:
            lpos[0] = cpos[0]
            lpos[1] = cpos[1]
            cpos[0] = x
            cpos[1] = y
            return cpos

    def get_state(self):
        return self.current_pos
    
    def get_color(self):
        return self.color

    # Helper Functions
    def inbounds(self, x, y):
        if 0 <= x < self.xs and 0 <= y < self.ys:
            return True
        return False

    def nactions(self):
        return len(self.eactions)

    def set_position(self, xs, ys):
        self.current_pos[0] = xs
        self.current_pos[1] = ys
    
    def set_disable_movement(self, bool):
        self.disable_movement = bool
        
    def set_disable_movement_steps(self, steps):
        self.disable_movement_steps = steps
        
    def get_disable_movement(self):
        return self.disable_movement

    def current_position(self):
        return self.current_pos

    def last_position(self):
        return self.last_pos
    
    def set_fuel(self, fuelAmount):
        self.currentFuel = fuelAmount
        
    def fuel_deplete(self):
        self.currentFuel -= 1
        
    def get_current_fuel(self):
        return self.currentFuel
    
    def get_fuel_diff(self):
        return self.fullFuel - self.currentFuel
    
    def make_fuel_full(self):
        self.currentFuel = self.fullFuel
        
    def get_team_nr(self):
        return self.team_nr
