import numpy as np 
import random as r
import TwoDEnv as Env

class nStepQLearningAgent:
    def __init__(self, env:Env.Env2d, n:int, gamma:float, maxIterations:int,epsilon:float)->None:
        self.env = env
        self.n = n
        self.g = gamma
        self.maxIterations = maxIterations
        self.e = epsilon
        self.e0 = self.e

        self.Qgrid = np.ones(shape=(4,self.env.rows,self.env.cols), dtype=float)
        self.pos = np.zeros(2, dtype=int) # agent's position
        self.actions = np.array(['←','↑','→','↓'])
        self.TPgrid = np.random.choice(a=['←','↑','→','↓'],size=(self.env.rows,self.env.cols))
        self.SVgrid = np.zeros(shape=(4,self.env.rows,self.env.cols))

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
                
    def act(self):
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

    def train(self,alpha:float,e_min:float,e_decay:float):
        print("Terminal states:", self.env.terminationStates)
        for episode in range(self.maxIterations):
            self.pos = np.array([r.randint(a=0,b=self.env.rows-1),r.randint(a=0,b=self.env.cols - 1)])
            self.e = max(e_min, self.e0 * (e_decay**episode))
            a = self.act()
            actions = [a]
            states = [self.pos.copy()]
            rewards = [0]

            T = float('inf')
            t = 0
            while True:
                if t < T:
                    reward = self.env.Rgrid[self.pos[0],self.pos[1]]
                    rewards.append(reward)
                    states.append(self.pos.copy())

                    if self.pos in self.env.terminationStates:
                        T = t+1
                    else:
                        actions.append(self.act())


                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n, T+1)):
                        G += (self.g ** (i - tau - 1)) * rewards[i]

                    if tau + self.n < T:
                        s_tau_n = states[tau+self.n]
                        a_tau_n = actions[tau + self.n]
                        G += (self.g ** self.n) * np.max(self.Qgrid[:,s_tau_n[0],s_tau_n[1]])

                    s_tau = states[tau]
                    a_tau = actions[tau]
                    self.SVgrid[a_tau,s_tau[0],s_tau[1]] += 1
                    old_q = self.Qgrid[a_tau,s_tau[0],s_tau[1]]
                    a = 1/(1+self.SVgrid[a_tau,s_tau[0],s_tau[1]])
                    self.Qgrid[a_tau,s_tau[0],s_tau[1]] += a * (G - old_q)
                    self.TPgrid[s_tau[0],s_tau[1]] = self.highestValueAction(s_tau)

                if tau == T - 1:
                    break
                t+=1

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.maxIterations}")
                print("Policy grid:")
                print(self.TPgrid)

#driver code
env = Env.Env2d(10,10,4)
print(env.Rgrid)
print(type(env.terminationStates))
agent = nStepQLearningAgent(env=env,n=3,gamma=0.9,maxIterations=5000,epsilon=0.25)
agent.train(0.3,0.01,0.99)
