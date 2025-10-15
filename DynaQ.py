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

        self.Qgrid = np.randomvarA__A = np.zeros((4, self.env.rows, self.env.cols)) 
        self.Pgrid = np.random.choice(a=['←','↑','→','↓'],size=(self.env.rows,self.env.cols)) # Policy grid
        self.pos = np.zeros(2, dtype=int) # agent's position
        self.actions = np.array(['←','↑','→','↓'])
        self.TPgrid = self.Pgrid.copy()
        self.model = {}


    def highestValueAction(self,pos=None,returnStr=False):
        if pos is None:
            temp = self.Qgrid[:,self.pos[0],self.pos[1]]
            idx = np.argmax(temp)
        else:
            temp = self.Qgrid[:, pos[0],pos[1]]
            idx = np.argmax(temp)
        
        if returnStr:
            match(idx):
                case 0:
                    return '←'
                case 1:
                    return '↑'
                case 2:
                    return '→'
                case 3:
                    return '↓'
        else:
            return idx                
    def act(self) -> int:
        if r.random() < 1 - self.e:
            # Greedy action
            choice = self.highestValueAction(self.pos)
        else:
            # Random action
            choice = r.randint(0, 3)  # 0=←, 1=↑, 2=→, 3=↓
        
        # Move agent according to the action index
        if choice == 0: self.actL()
        elif choice == 1: self.actU()
        elif choice == 2: self.actR()
        elif choice == 3: self.actD()
        
        return choice

                
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
        print(self.env.terminationStates)
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

                self.TPgrid[prevPos[0],prevPos[1]] = self.highestValueAction(prevPos,True)
                self.model[(tuple(prevPos), a)] = (tuple(self.pos),self.env.Rgrid[self.pos[0], self.pos[1]])
                
                for i in range(self.n):
                    key = r.choice(list(self.model.keys()))
                    nextS, Reward = self.model[key]
                    nextA = self.highestValueAction(key[0])
                    self.Qgrid[key[1], key[0][0],key[0][1]] += alpha * (
                        Reward + self.g * self.Qgrid[nextA, nextS[0],nextS[1]] - self.Qgrid[key[1],key[0][0] ,key[0][1]]
                    )

                if tuple(self.pos) in self.env.terminationStates:
                    break


            if episode%100 ==0:
                print(f'{episode }/{self.maxIterations} = \n {self.TPgrid}')

#driver code
env = Env.Env2d(10,10,4)
agent = QLearningAgent(env=env, gamma=0.9,epsilon=0.3,maxIterations=5000,planningSteps=5)
agent.train(0.2)