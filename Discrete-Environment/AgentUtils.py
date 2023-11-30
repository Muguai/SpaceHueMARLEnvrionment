import numpy as np

from DiscreteAgent import DiscreteAgent

#################################################################
# Implements utility functions for multi-agent DRL
#################################################################


def create_agents(
    nagents,
    map_matrix,
    obs_range,
    randomizer,
    flatten=False,
    randinit=False,
    constraints=None,
):
    """Initializes the agents on a map (map_matrix).
    -nagents: the number of agents to put on the map
    -randinit: if True will place agents in random, feasible locations
               if False will place all agents at 0
    expanded_mat: This matrix is used to spawn non-adjacent agents
    """
    #availableCols = [(255,0,0), (0,255,0), (0,0,255),(255,255,0), (0,255,255), (255,0,255),(128,0,0), (0,128,0), (0,0,128), (128,128,0)]
    xs, ys = map_matrix.shape
    availableCols = [
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
    agents = []
    expanded_mat = np.zeros((xs + 2, ys + 2))
    for i in range(nagents):
        xinit, yinit = (0, 0)
        if randinit:
            print("rand")
            xinit, yinit = feasible_position_exp(
                randomizer, map_matrix, expanded_mat, constraints=constraints
            )
            # fill expanded_mat
            expanded_mat[xinit + 1, yinit + 1] = -1
            expanded_mat[xinit + 2, yinit + 1] = -1
            expanded_mat[xinit, yinit + 1] = -1
            expanded_mat[xinit + 1, yinit + 2] = -1
            expanded_mat[xinit + 1, yinit] = -1
        else:
            print("not Rand", ys)
            print(i + ((ys // 2) - (nagents // 2)))
            yinit = i + ((ys // 2) - (nagents // 2))
            
        agent = DiscreteAgent(
            xs, ys, map_matrix, randomizer, obs_range=obs_range, flatten=flatten, col=availableCols[i]
        )
        if(xinit != 0):
            xinit = xinit / 2;
        agent.set_position(xinit, yinit)
        agents.append(agent)
    return agents


def feasible_position_exp(randomizer, map_matrix, expanded_mat, constraints=None):
    """Returns a feasible position on map (map_matrix)."""
    xs, ys = map_matrix.shape
    while True:
        if constraints is None:
            x = randomizer.integers(0, xs)
            y = randomizer.integers(0, ys)
        else:
            xl, xu = constraints[0]
            yl, yu = constraints[1]
            x = randomizer.integers(xl, xu)
            y = randomizer.integers(yl, yu)
        if map_matrix[x, y] != -1 and expanded_mat[x + 1, y + 1] != -1:
            return (x, y)


def set_agents(agent_matrix, map_matrix):
    # check input sizes
    if agent_matrix.shape != map_matrix.shape:
        raise ValueError("Agent configuration and map matrix have mis-matched sizes")

    agents = []
    xs, ys = agent_matrix.shape
    for i in range(xs):
        for j in range(ys):
            n_agents = agent_matrix[i, j]
            if n_agents > 0:
                if map_matrix[i, j] == -1:
                    raise ValueError(
                        "Trying to place an agent into a building: check map matrix and agent configuration"
                    )
                agent = DiscreteAgent(xs, ys, map_matrix)
                agent.set_position(i, j)
                agents.append(agent)
    return agents