import numpy as np
import random as r
import TwoDEnv as Env

class QLearningAgent:
    def __init__(self, env:Env, gamma:float, epsilon:float, maxIterations: int, lambd:float)->None:
        self.env = env
        self.g = gamma
        self.e0 = epsilon
        self.e = self.e0
        self.maxIterations = maxIterations
        self.l = lambd

        self.Qgrid = np.random.randint(low=-2,high = 2,size=(4, self.env.rows, self.env.cols )).astype(float) # Q function grid
        self.Pgrid = np.random.choice(a=['←','↑','→','↓'],size=(self.env.rows,self.env.cols)) # Policy grid
        self.pos = np.zeros(2, dtype=int) # agent's position
        self.actions = np.array(['←','↑','→','↓'])
        self.Egrid = np.zeros_like(self.Qgrid) # eligibility trace grid


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
            self.e = max(e_min,self.e0 * (e_decay**episode))
            self.pos= np.array([r.randint(0,self.env.rows-1),r.randint(0,self.env.cols-1)])
            s = self.pos.copy()
            a = self.act()
            self.Egrid[:,:,:] = 0
            while True:
                s_next = self.pos.copy()
                a_next = self.act()
                delta = self.env.Rgrid[s_next[0],s_next[1]] + self. g * self.Qgrid[a_next,s_next[0],s_next[1]] - self.Qgrid[a,s[0],s[1]]
                self.Egrid[a,s[0],s[1]] = 1

                for i,j,k in np.ndindex(self.Qgrid.shape):
                    self.Qgrid[i,j,k] += alpha*delta*self.Egrid[i,j,k]
                    self.Egrid[i,j,k] = self.g*self.l * self.Egrid[i,j,k]
                    self.Pgrid[j,k] = self.highestValueAction([j,k])
                s = s_next.copy()
                a = a_next

                if self.pos in self.env.terminationStates:
                    break

            if (episode+1)%100 == 0:
                print(f'{episode + 1}/{self.maxIterations} = \n {self.Pgrid}')

#driver code
env = Env.Env2d(10,10,4)
agent = QLearningAgent(env, 0.9,0.,5000,0.5)
print(env.terminationStates)
agent.train(0.5)