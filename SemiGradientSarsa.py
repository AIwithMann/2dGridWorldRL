import random as r 
import numpy as np
from ContinuousTwoDEnv import ContinuousEnv2d as ENV

class SemiGradientSarsa:
    def __init__(self, env:ENV, alpha:float,gamma:float, epsilon:float, maxIterations:int,delta:int)->None:
        self.env = env
        self.alpha = alpha
        self.g = gamma
        self.e = epsilon
        self.d = 3
        self.maxIter = maxIterations
        self.delta = delta

        self.w = np.zeros(self.d)

        self.actions={
            0:self.actL,
            1:self.actU,
            2:self.actR,
            3:self.actD
        }
        #no q grid because we are using function approximation
        # instead, we'll use feature vector, but the problem is, how may I construct the feature vector?
        #let us just use the x and y coordinates of the state as the feature vector
        # Then we won't need tile coding or other such methods

    def Q(self,pos:np.ndarray=None,action=None)->float:
        return np.dot(self.w,np.array([pos[0],pos[1],action]))

#--------------------------------------------------------------------------------------------------------------------------------
    def actL(self, pos): return np.array([pos[0], max(0, pos[1] - self.delta)])
    def actU(self, pos): return np.array([max(0, pos[0] - self.delta), pos[1]])
    def actR(self, pos): return np.array([pos[0], min(self.env.yInterval - 1e-5, pos[1] + self.delta)])
    def actD(self, pos): return np.array([min(self.env.xInterval - 1e-5, pos[0] + self.delta), pos[1]])
            
#--------------------------------------------------------------------------------------------------------------------------------

    def highestValueAction(self,pos:np.ndarray):
        actionValues = [self.Q(self.actions[a](pos), a) for a in range(4)]
        return max(range(4), key=lambda a: actionValues[a])

    def act(self,pos):
        prob = r.random()
        if prob > self.e:
            action = self.highestValueAction(pos)
        else:
            action = r.choice([0,1,2,3])
        newPos = self.actions[action](pos)
        return action, newPos

    def learn(self):
        for episode in range(self.maxIter):
            print(episode, end='\n')
            pos = np.array([r.uniform(a=0,b=self.env.xInterval),r.uniform(a=0,b=self.env.yInterval)])
            a,nextS = self.act(pos)
            
            while True:
                reward = self.env.reward(pos[0],pos[1])
                phi = np.array([pos[0],pos[1],a])
                if any(np.allclose(pos, t) for t in self.env.terminationStates):
                    self.w += self.alpha * (reward - self.Q(pos,a)) * phi
                    break 

                nextA,nextNextS = self.act(nextS)
                target = reward + self.g*self.Q(nextS,nextA)
                self.w += self.alpha * (target- self.Q(pos,a))*phi

                pos, a, nextS = nextS, nextA, nextNextS

            print(episode,"\n")
#driver code
env = ENV(10,10,10)
agent = SemiGradientSarsa(env,0.01,0.9,0.25,1000,1.1)
agent.learn()
                    
