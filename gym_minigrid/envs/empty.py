from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        # agent_start_pos=(1,1),
        agent_start_pos=agent_start_pos_def, # toshiki
        agent_goal_pos=agent_goal_pos_def, # toshiki
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_goal_pos = agent_goal_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4*size*size, # what's this?? (toshiki)
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Place a goal square in the bottom-right corner
        # self.put_obj(Goal(), width-2+self.agent_goal_pos[0], height-2+self.agent_goal_pos[1]) # toshiki

        goal_width = 1
        for i in range(0, goal_width):
            for j in range(0, goal_width):
                self.put_obj(Goal(), width-2+self.agent_goal_pos[0]-i, height-2+self.agent_goal_pos[1]-j) # toshiki

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

class EmptyEnv5x5(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)

class EmptyRandomEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class EmptyEnv6x6(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, **kwargs)

class EmptyRandomEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)

class EmptyEnv16x16(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)

class EmptyEnv32x32(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=32, **kwargs)

class EmptyEnv52x52(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=52, **kwargs)

class EmptyEnv102x102(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=102, **kwargs)

class EmptyEnv142x142(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=142, **kwargs)

class EmptyEnv202x202(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=202, **kwargs)

register(
    id='MiniGrid-Empty-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5'
)

register(
    id='MiniGrid-Empty-Random-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv5x5'
)

register(
    id='MiniGrid-Empty-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyEnv6x6'
)

register(
    id='MiniGrid-Empty-Random-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv6x6'
)

register(
    id='MiniGrid-Empty-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnv'
)

register(
    id='MiniGrid-Empty-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyEnv16x16'
)

register(
    id='MiniGrid-Empty-32x32-v0',
    entry_point='gym_minigrid.envs:EmptyEnv32x32'
)

register(
    id='MiniGrid-Empty-52x52-v0',
    entry_point='gym_minigrid.envs:EmptyEnv52x52'
)

register(
    id='MiniGrid-Empty-102x102-v0',
    entry_point='gym_minigrid.envs:EmptyEnv102x102'
)

register(
    id='MiniGrid-Empty-142x142-v0',
    entry_point='gym_minigrid.envs:EmptyEnv142x142'
)

register(
    id='MiniGrid-Empty-202x202-v0',
    entry_point='gym_minigrid.envs:EmptyEnv202x202'
)
