from Discrete_Environment.Raw_Discrete import env

# more tests available at https://pettingzoo.farama.org/content/environment_tests/
from pettingzoo.test import api_test

env = env(
    render_mode="human",
    x_size=32,
    y_size=16,
    max_cycles=1000,
    randomActions=True,
    randomActionsProbability=0.2,
    randomSpawn=True,
    sparseReward=False,
    fullyObservable=False,
    competitive=False,
    moveTime=3,
    spawnTime=1,
    obstacle_probability=0.5,
    num_obstacles=2,
    n_agents=4,
    obs_range=7
)

api_test(env, num_cycles=1000)