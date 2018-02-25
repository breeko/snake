import numpy as np
from array2gif import write_gif
from IPython.display import Image
import gym
from gym.spaces.discrete import Discrete
from gym.utils import seeding

UP    = (-1, 0)
RIGHT = ( 0, 1)
DOWN  = ( 1, 0)
LEFT  = ( 0,-1)

action_2_move = {
    0: UP,
    1: RIGHT,
    2: DOWN,
    3: LEFT
}

class Obstacle:
    def __init__(self, coors):
        if type(coors) is not list:
            coors = [coors]
        self.coors = coors

class Food(Obstacle):
    def __init__(self, coors):
        super().__init__(coors)
        self.color = [0, 1., 0]
    
class Snake(Obstacle):
    def __init__(self, coors):
        super().__init__(coors)
        self.grow = False
        self.color = [0 , 0 , 1.]
        self.direction = action_2_move[np.random.choice(len(action_2_move))]
        self.viewer = None
        
    def step(self, move):
        assert move in action_2_move.values(), "Invalid move"
        
        if self.direction[0] + move[0] == 0 and self.direction[1] + move[1] == 0:
            # Try to move in opposite direction, continue in original direction. 
            # E.g. moving RIGHT and try to move LEFT.
            move = self.direction
            
        new_pos = (self.coors[0][0] + move[0], self.coors[0][1] + move[1])
        self.coors.insert(0,new_pos)
        if not self.grow:
            self.coors = self.coors[:-1]
            
        self.direction = move
        self.grow = False
        
class Game(gym.Env):
    def __init__(self, PlayerType=Snake, height=20, width=20):
        self.height = height
        self.width = width
        self.action_space = Discrete(len(action_2_move.keys()))
        self.actions = self.action_space.n
        self.reset(PlayerType)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
	
    def reset(self, PlayerType=Snake):
        self.done = False
        self.frames = []
        self.add_player(PlayerType)
        self.add_food()
        self._update_state()
        return self.state
    
    def _random_coors(self, border=0, exclude=[]):
        x = np.random.randint(border, self.width - border)
        y = np.random.randint(border, self.height - border)
        if (y,x) in exclude:
            return self._random_coors(exclude=exclude)
        return (y, x)
    
    def step(self, action):
        reward = 0
        if not self.done:
            move = action_2_move[action]
            self.player.step(move)

            if self._check_out_of_bounds():
                reward = -1
                self.done = True
            elif self._check_intersection():
                reward = -1
                self.done = True
            elif self._check_eat():
                reward = 1
                self.add_food()
                self.player.grow = True

            self._update_state()
        
        return self.state, reward, self.done, {}
    
    def _check_out_of_bounds(self):
        head = self.player.coors[0]
        return head[1] < 0 or head[1] >= self.width or head[0] < 0 or head[0] >= self.height
    
    def _check_intersection(self):
        return self.player.coors[0] in self.player.coors[1:]
    
    def _check_eat(self):
        return self.food.coors[0] == self.player.coors[0]
    
    def add_player(self, PlayerType):
        coors = self._random_coors(border=1)		
        self.player = PlayerType(coors)

    def add_food(self):
        coors = self._random_coors(exclude=self.player.coors)
        self.food = Food(coors)
    
    def _update_state(self):
        self.state = np.zeros((self.height, self.width, 3))
        if not self.done:
            for pos in self.player.coors:
                try:
                    self.state[pos[0]][pos[1]] = self.player.color
                except IndexError:
                    pass

        for pos in self.food.coors:
            self.state[pos[0]][pos[1]] = self.food.color
        
        self.frames.append(self.state)
        
    def render(self,mode="human"):
        return self.state
		
    def close(self):
        if self.viewer: self.viewer.close()
    
    def save(self, filename="out.gif", size=(250,250)):
        mult = min(size[0] // self.height, size[1] // self.width)
        
        frames = np.copy(self.frames)
        frames = np.repeat(frames,mult,axis=1)
        frames = np.repeat(frames,mult,axis=2)
        
        frames = np.rollaxis(frames,-1,1)
        frames *= 255
        frames = frames.astype(int)
        write_gif(list(frames), filename, fps=5)
        return Image(filename)