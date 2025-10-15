import numpy as np
import random as r
import TwoDEnv as Env

class QLearningAgent:
    def __init__(self, env:Env, gamma:float, epsilon:float,planningSteps:int, maxIterations: int)->None:
        self.env = env
        self.g = gamma
        self.e0 = epsilon
        self.e = self.e0
        self.maxIterations = maxIterations
        self.n = planningSteps

        self.Qgrid = np.random.randint(low=-2,high = 2,size=(4, self.env.rows, self.env.cols )).astype(float) # Q function grid
        self.Pgrid = np.random.choice(a=['←','↑','→','↓'],size=(self.env.rows,self.env.cols)) # Policy grid
        self.pos = np.zeros(2, dtype=int) # agent's position
        self.actions = np.array(['←','↑','→','↓'])
        self.TPgrid = self.Pgrid.copy()
        self.model = {}


    def highestValueAction(self,pos=None):
        if pos is None:
            temp = self.Qgrid[:,self.pos[0],self.pos[1]]
            return np.argmax(temp)
        else:
            temp = self.Qgrid[:, pos[0],pos[1]]
            idx = np.argmax(temp)
            match(idx):
                case 0:
                    return '←'
                case 1:
                    return '↑'
                case 2:
                    return '→'
                case 3:
                    return '↓'
                
    def act(self)->int:
        
        pr = 1 - self.e 
        if r.random()<pr:
            choice = self.highestValueAction(self.pos)
            match(choice):
                case '←':
                    self.actL()
                    return 0
                case '↑':
                    self.actU()
                    return 1
                case '→':
                    self.actR()
                    return 2
                case '↓':
                    self.actD()
                    return 3
            return self.actions.tolist().index(choice)
        else:
            choice = r.choice(['←','↑','→','↓'])
            match(choice):
                case '←':
                    self.actL()
                case '↑':
                    self.actU()
                case '→':
                    self.actR()
                case '↓':
                    self.actD()
            return self.actions.tolist().index(choice)
                
    def actL(self)->None: # Moving Left
        if self.pos[1]==0:
            pass
        else:
            self.pos[1]-=1
    def actR(self)->None: # Moving Right
        if self.pos[1] == self.env.cols-1:
            pass
        else:
            self.pos[1] += 1
    
    def actU(self)->None: # Moving Up
        if self.pos[0]==0:
            pass
        else:
            self.pos[0]-=1
    
    def actD(self)->None: # Moving Down
        if self.pos[0] == self.env.rows - 1:
            pass
        else:
            self.pos[0] += 1

    def train(self,alpha:float,e_min:int = 0.01, e_decay:float = 0.995)->None:
        for episode in range(self.maxIterations):
            t = 0
            self.e = max(e_min, self.e0 * (e_decay**episode))
            row = r.randint(0,self.env.rows - 1)
            col = r.randint(0, self.env.cols - 1)
            self.pos = np.array([row,col])

            while True:
                prevPos = self.pos.copy()
                a = self.act()
                nextA = self.highestValueAction()
                self.Qgrid[a, prevPos[0], prevPos[1]] += alpha * (
                    self.env.Rgrid[self.pos[0], self.pos[1]] 
                    + self.g * self.Qgrid[nextA, self.pos[0], self.pos[1]] 
                    - self.Qgrid[a, prevPos[0], prevPos[1]]
                )   

                self.TPgrid[prevPos[0],prevPos[1]] = self.highestValueAction(prevPos)
                self.model[]
                if self.pos in self.env.terminationStates:
                    t = 1
                elif t ==1:
                    break


            if (episode+1)%100 == 0:
                print(f'{episode + 1}/{self.maxIterations} = \n {self.TPgrid}')

#driver code
env = Env.Env2d(10,10,4)
agent = QLearningAgent(env, 0.9,0.3,5000)
agent.train(0.5)