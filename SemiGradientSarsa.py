import random as r 
import numpy as np
from ContinuousTwoDEnv import ContinuousEnv2d as ENV

class SemiGradientSarsa:
    def __init__(self, env:ENV, alpha:float, epsilon:float, maxIterations:int,delta:int)->None:
        self.env = env
        self.alpha = alpha
        self.e = epsilon
        self.d = 3
        self.maxIter = maxIterations
        self.delta = delta

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
        if pos is not None:
            return np.dot(self.w,np.array(pos[0],pos[1],action))
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

    def highestValueAction(self,pos:np.ndarray=None):
        if pos is not None:
            actionValues = [self.Q(f(pos,True)) for f in [self.actL,self.actU,self.actR,self.actD]]
            return max(enumerate(actionValues),key= lambda x:x[1])[0]
        else:
            actionValues = [self.Q(f(plan=True)) for f in [self.actL,self.actU,self.actR,self.actD]]
            return max(enumerate(actionValues),key= lambda x:x[1])[0]
    
    def act(self,pos:np.ndarray=None):
        prob = r.random()
        if prob > self.e:
            action = self.highestValueAction(pos=pos) if pos is not None else self.highestValueAction()
            self.actions[action](pos=pos) if pos is not None else self.actions[action]()
            return action
        else:
            action = r.sample(population=[0,1,2,3],k=1)
            self.actions[action](pos=pos) if pos is not None else self.actions[action]()
            return action
    
    def learn(self):
        for episode in range(self.maxIter):
            self.pos = np.array([r.uniform(a=0,b=self.env.xInterval),r.uniform(a=0,b=self.env.yInterval)])
            a = self.act()
            while True:
                if any((np.round(x) == np.round(xs)) and (np.round(y) == np.round(ys)) for xs, ys in self.terminationStates): 
                    self.w[0] += self.alpha * (self.env.reward(self.pos[0],self.pos[1]) - self.Q())*np.array()