import numpy as np
import random as r
import TwoDEnv as Env

class SarsaAgent:
    def __init__(self, env:Env, gamma:float, epsilon:float,maxIterations:int,numBackups:int)->None:
        self.env = env
        self.g = gamma
        self.e = epsilon
        self.maxIterations = maxIterations
        self.n = numBackups

        self.Qgrid = np.random.randint(low=-2,high = 2,size=(4, self.env.rows, self.env.cols )).astype(float) # Q function grid
        self.Pgrid = np.random.choice(a=['←','↑','→','↓'],size=(self.env.rows,self.env.cols)) # Policy grid
        self.pos = np.zeros(2, dtype=int) # agent's position
        self.actions = np.array(['←','↑','→','↓'])
        
        term_rows, term_cols = self.env.terminationStates
        self.terminationStates = list(zip(term_rows, term_cols))

    def highestValueAction(self, pos=None):
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
            choice = self.Pgrid[self.pos[0],self.pos[1]]
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
            
    def train(self,alpha:float):
        print("Terminal states:", self.terminationStates)
        for episode in range(self.maxIterations): 

            row = np.random.randint(low=0, high=self.env.rows)
            col = np.random.randint(low=0, high=self.env.cols)
            self.pos = np.array([row, col])

            action_idx = self.act()

            states = [self.pos.copy()]
            actions = [action_idx]
            rewards = [0] 

            T = float('inf')
            t = 0
            while True:
                if t < T:
                    reward = self.env.Rgrid[self.pos[0], self.pos[1]]
                    rewards.append(reward)
                    states.append(self.pos.copy())

                    if tuple(self.pos) in self.terminationStates:
                        T = t + 1
                    else:
                        action_idx = self.act() 
                        actions.append(action_idx)
                
                tau = t - self.n + 1 # tau is the time whose state's value is being updated
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += (self.g ** (i - tau - 1)) * rewards[i]
                    
                    if tau + self.n < T:
                        s_tau_n = states[tau + self.n]
                        a_tau_n = actions[tau + self.n]
                        G += (self.g ** self.n) * self.Qgrid[a_tau_n, s_tau_n[0], s_tau_n[1]]
                    
                    s_tau = states[tau]
                    a_tau = actions[tau]
                    
                    old_q = self.Qgrid[a_tau, s_tau[0], s_tau[1]]
                    self.Qgrid[a_tau, s_tau[0], s_tau[1]] += alpha * (G - old_q)
                    self.Pgrid[s_tau[0], s_tau[1]] = self.highestValueAction(s_tau)

                if tau == T - 1:
                    break
                
                t += 1
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.maxIterations}")
                print("Policy grid:")
                print(self.Pgrid)


#driver code

env = Env.Env2d(5,5,2)
agent = SarsaAgent(env,0.9,0.3,1000,5)
agent.train(0.5)
