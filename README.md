# Space Hue MARL Environment

This project provides a flexible environment for Multi-Agent Reinforcement Learning (MARL) called Space Hue. It allows you to customize various parameters to suit your specific needs.

## Environment Parameters

- `render_mode`: The mode for rendering the environment. Default is "human".
- `x_size`: The width of the environment grid. Default is 32.
- `y_size`: The height of the environment grid. Default is 16.
- `max_cycles`: The maximum number of cycles the environment runs for. Default is 1000.
- `randomActions`: Whether agents has a chance to do a random action other then the one chosen. Default is False.
- `randomActionsProbability`: The probability of a random action occuring if randomActions is enabled. Default is 0.2.
- `randomSpawn`: Whether agents spawn at random locations and wheter obstacles spawn randomly or in a pre determined pattern. Default is True.
- `sparseReward`: Sparse reward mode. if on agents wont gain a small reward each step and instead gain reward when they hit fuel obstacle. Default is False.
- `fullyObservable`: Whether the environment is fully observable. Default is False.
- `competitive`: Whether the environment is competitive. Default is False.
- `moveTime`: The time it takes for an agent to move. Default is 3.
- `spawnTime`: The time it takes for an agent to spawn. Default is 1.
- `num_obstacles`: The number of obstacles created at a time in the environment.
- `obstacle_probability`: The probability of an obstacle or wall being spawned in the environment. The value is clamped between 0 and 1, meaning it can't be less than 0 or more than 1. Only affects env if `randomSpawn` is True
- `n_agents`: The number of agents in the environment. Default is 4.
- `obs_range`: The range of observation for an agent. Default is 7.

## How to Run

1. Ensure you have Python 3.9 installed on your machine. You can download it from the official Python website.
2. Clone this repository to your local machine.
3. Navigate to the project directory in your terminal.
4. Install the required dependencies using pip:

```python
pip install -r requirements.txt
```

5. Run the test script:

```python
python test.py
```

## Gif

![spacehue](https://github.com/Muguai/SpaceHueMARLEnvrionment/assets/37656342/89887fff-f370-49b7-b588-7e31741a3c65)



