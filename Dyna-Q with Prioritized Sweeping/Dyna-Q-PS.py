import numpy as np
import random as r
import sys
sys.path.append("/home/username/Forever-Beta/2dGridWorldRL")
import TwoDEnv as Env

from PriorityQ import Queue

class QLearningAgent:
    def __init__(self, env:Env, gamma:float, epsilon:float,planningSteps:int, maxIterations: int)->None:
        self.env = env
        self.g = gamma
        self.e0 = epsilon
        self.e = self.e0
        self.maxIterations = maxIterations
        self.n = planningSteps
        self.Q = Queue()
        self.Qgrid = np.zeros((4, self.env.rows, self.env.cols)) 
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
            match idx :
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
#------------------------------------------------------------------------------------------------------
                
    def actL(self)->None: # Moving Left
        if self.pos[1]==0: pass
        else:
            self.pos[1]-=1
    def actR(self)->None: # Moving Right
        if self.pos[1] == self.env.cols-1: pass
        else:
            self.pos[1] += 1
    
    def actU(self)->None: # Moving Up
        if self.pos[0]==0: pass
        else:
            self.pos[0]-=1
    
    def actD(self)->None: # Moving Down
        if self.pos[0] == self.env.rows - 1: pass
        else:
            self.pos[0] += 1

#------------------------------------------------------------------------------------------------------

    def preL(self,state):
        if state[1]==0: return None

        sa = ((state[0], state[1]-1), 0)
        if sa not in self.model.keys(): return None
        return sa
        
    def preU(self,state):
        if state[0]==0: return None
        
        sa = ((state[0]-1,state[1]),1)
        if sa not in self.model.keys(): return None
        return sa
    
    def preR(self,state):
        if state[1] == self.env.cols-1: return None

        sa = ((state[0],state[1]+1),2)
        if sa not in self.model.keys(): return None
        return sa

    def preD(self,state):
        if state[0] == self.env.rows-1: return None

        sa = ((state[0]+1,state[1]),3)
        if sa not in self.model.keys(): return None
        return sa

#------------------------------------------------------------------------------------------------------

    def findPredecessors(self,state:np.ndarray):
        states = []
        states.append(self.preL(state))
        states.append(self.preU(state))
        states.append(self.preR(state))
        states.append(self.preD(state))
        return states
    def train(self,alpha:float,e_min:int = 0.01, e_decay:float = 0.995,theta:float = 10e-3)->None:
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
                P = abs(self.env.Rgrid[self.pos[0], self.pos[1]] 
                    + self.g * self.Qgrid[nextA, self.pos[0], self.pos[1]] 
                    - self.Qgrid[a, prevPos[0], prevPos[1]]
                )
                self.model[(tuple(prevPos), a)] = (tuple(self.pos),self.env.Rgrid[self.pos[0], self.pos[1]])
                if P > theta:
                    self.Q.enqueue((P,tuple(prevPos),a))

                for i in range(self.n):
                    item = self.Q.dequeue()
                    if item is None: break
                    P, S, A = item

                    nextS, R = self.model[(S, A)]
                    nextA = self.highestValueAction(nextS)
                    self.Qgrid[A, S[0], S[1]] += alpha * (
                        R 
                        + self.g * self.Qgrid[nextA, nextS[0], nextS[1]] 
                        - self.Qgrid[A, S[0], S[1]]
                    )
                    self.TPgrid[S[0],S[1]] = self.highestValueAction(S,True)
                    for sa in self.findPredecessors(np.array(S)):
                        if sa == None:
                            continue
                        _, Reward = self.model[sa]
                        P = abs(Reward + self.g * self.Qgrid[A,S[0],S[1]] - self.Qgrid[sa[1],sa[0][0],sa[0][1]])
                        if P > theta:
                            self.Q.enqueue((P,sa[0],sa[1]))
                    

                if tuple(self.pos) in self.env.terminationStates:
                    break


            if episode%10 ==0:
                print(f'{episode }/{self.maxIterations} = \n {self.TPgrid}')

#driver code
env = Env.Env2d(10,10,4)
agent = QLearningAgent(env=env, gamma=0.9,epsilon=0.3,maxIterations=5000,planningSteps=5)
agent.train(0.2,0.01,0.995,10e-3)