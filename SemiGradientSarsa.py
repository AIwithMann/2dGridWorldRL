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

        self.pos = np.zeros(2)
        self.action = -1
        self.features = np.zeros(self.d)
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
        if pos is not None and action is not None:
            return np.dot(self.w,np.array([pos[0],pos[1],action]))
        return np.dot(self.w,self.features)

#--------------------------------------------------------------------------------------------------------------------------------
    def actL(self,pos:np.ndarray=None,plan:bool=False):
        y = self.pos[1]-1
        if y<0 and plan:
            return np.array([self.pos[0],0])
        elif y<0 and not plan:
            self.features[1] = 0
        elif y>=0 and not plan:
            return np.array([self.pos[0],y])
        else:
            self.features[1] = y

    def actU(self,pos:np.ndarray=None,plan:bool=False):
        x = self.pos[0]-1
        if x<0 and plan:
            return np.array([0,self.pos[1]])
        elif x<0 and not plan:
            self.features[0] = 0
        elif x>=0 and plan:
            return np.array([x,self.pos[1]])
        else:
            self.features[0] = x

    def actR(self,pos:np.ndarray=None,plan:bool=False):
        y = self.features[1]+self.delta
        if y>=self.env.yInterval and plan:
            return np.array([self.pos[0],self.env.yInterval-(10e-5)])
        elif y>=self.env.yInterval and not plan:
            self.features[1] = self.env.yInterval-(10e-5)
        elif y<self.env.yInterval and plan:
            return np.array([self.pos[0],y])
        else:
            self.features[1] = y

    def actD(self,pos:np.ndarray=None,plan:bool=False):
        x = self.features[0]+self.delta
        if x>=self.env.xInterval and plan:
            return np.array([self.env.xInterval-(10e-5),self.pos[1]])
        elif x>=self.env.xInterval and not plan:
            self.features[0] = self.env.xInterval-(10e-5)
        elif x<self.env.xInterval and plan:
            return np.array([x,self.features[1]])
        else:
            self.features[0] = x
            
#--------------------------------------------------------------------------------------------------------------------------------

    def highestValueAction(self):
        actionValues = [self.Q(f(plan=True)) for f in [self.actL,self.actU,self.actR,self.actD]]
        return max(enumerate(actionValues),key= lambda x:x[1])[0]

    def act(self):
        prob = r.random()
        if prob > self.e:
            action = self.highestValueAction()
            self.actions[action]()
        else:
            action = r.choice([0,1,2,3])
            self.actions[action]()
        return action

    def learn(self):
        for episode in range(self.maxIter):
            self.pos = np.array([r.uniform(a=0,b=self.env.xInterval),r.uniform(a=0,b=self.env.yInterval)])
            a = self.act()
            self.features[:] = self.pos[0], self.pos[1], a
            
            while True:
                if any((np.round(self.features[0]) == np.round(xs)) and (np.round(self.features[1]) == np.round(ys)) for xs, ys in self.env.terminationStates): 
                    self.w += self.alpha * (self.env.reward(self.features[0],self.features[1]) - self.Q()) * self.features
                    break 
                nextA = self.act()
                self.w += self.alpha * (self.env.reward(self.features[0],self.features[1]) + self.g *self.Q(self.pos,nextA) - self.Q())*self.features
                self.features[:] = self.pos[0],self.pos[1],nextA
            print(episode,"\n")
#driver code
env = ENV(10,10,4)
agent = SemiGradientSarsa(env,0.3,0.9,0.25,1000,1)
agent.learn()
                    
