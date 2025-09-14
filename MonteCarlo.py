import numpy as np
import random as r
import TwoDEnv as Env
    
class MCagent:
    def __init__(self, env:Env, gamma:float, epsilon:float, maxIterations:int)->None:
        self.env = env
        self.g = gamma
        self.e = epsilon
        self.maxIterations = maxIterations

        self.Vgrid = np.random.uniform(low=-2, high=2, size=(env.rows, env.cols))  # value function grid
        self.TPgrid = np.random.choice( a=['←','↑','→','↓'],size=(env.rows, env.cols)) # Target policy grid. Target policy is deterministic 
        self.BP = {
            "←":0.25,
            "↑":0.25,
            "→":0.25,
            "↓":0.25
        }
        
        self.pos = np.random.randint(low=0, high=env.rows, size=(2))

        self.actions = np.array(['←','↑','→','↓'])

        self.G_t = 0
        self.W_t = 1
        self.C_t = 1

        self.stateTrajectory = []
        self.rewardTrajectory = []
        self.actionTrajectory = []
    
    def highestValueAction(self)->str:
        actions = {
            0:'←',
            1:'↑',
            2:'→',
            3:'↓'
        }

        row, col = self.pos[0], self.pos[1]

        # if conditions are necessary to prevent the agent from going out of the grid
        l = self.Vgrid[row, col-1] if col != 0 else -np.inf
        u = self.Vgrid[row-1, col] if row != 0 else -np.inf
        r = self.Vgrid[row, col+1] if col != (self.env.cols-1) else -np.inf
        d = self.Vgrid[row+1, col] if row != (self.env.rows-1) else -np.inf


        arr = np.array([l,u,r,d])
        idx= np.argmax(arr)

        return actions[idx]

    def act(self, updating:bool)->str:
        pr =  1 - self.e #Probability for being greedy
        if r.random() < pr:
            choice = self.highestValueAction()
            match(choice):
                case '←':
                    self.actL()
                case '↑':
                    self.actU()
                case '→':
                    self.actR()
                case '↓':
                    self.actD()
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

        #Adding state, reward and action to their trajectories, so they can be used for policy ieration
        self.stateTrajectory.append(self.pos.copy())
        self.rewardTrajectory.append(self.env.Rgrid[self.pos[0], self.pos[1]])
        self.actionTrajectory.append(choice)

        return choice
    def generateEpisode(self)->None:
        termination_coords = list(zip(*self.env.terminationIndices))
        while tuple(self.pos) not in termination_coords:
            self.act(updating = False)

    def train(self):
        print("Terminal states:", list(zip(*self.env.terminationIndices)))
        for j in range(self.maxIterations):
            # reset episode
            self.stateTrajectory.clear()
            self.rewardTrajectory.clear()
            self.actionTrajectory.clear()
            self.pos = np.random.randint(low=0, high=self.env.rows, size=(2))

            self.generateEpisode()

            # backward update
            self.G_t = 0
            self.W_t = 1
            self.C_t = 1

            for i in range(len(self.stateTrajectory) - 1, -1, -1):
                self.G_t = self.g * self.G_t + self.rewardTrajectory[i]
                self.C_t += self.W_t
                self.Vgrid[self.stateTrajectory[i]] += (
                    self.W_t / self.C_t * (self.G_t - self.Vgrid[self.stateTrajectory[i]])
                )
                self.TPgrid[self.stateTrajectory[i]] = self.highestValueAction()
                # target policy is greedy: prob = 1 if action matches, else 0
                greedy_action = self.highestValueAction()
                if self.actionTrajectory[i] == greedy_action:
                    self.W_t *= 1.0 / self.BP[self.actionTrajectory[i]]
                else:
                    self.W_t = 0
                if self.W_t == 0:
                    break

            print("Value Grid:")
            print(self.Vgrid)
            print("Policy Grid:")
            print(self.TPgrid)

    
    def actL(self): #moving to left
        if self.pos[1] == 0:
            pass
        else:
            self.pos[1] -= 1
    
    def actR(self):#moving to right
        if self.pos[1] == self.env.cols-1:
            pass
        else:
            self.pos[1] += 1
    
    def actU(self): #moving up
        if self.pos[0] == 0:
            pass
        else:
            self.pos[0] -= 1

    def actD(self): #moving down
        if self.pos[0] == self.env.rows-1:
            pass
        else:
            self.pos[0] += 1

#Driver code
env = Env.Env2d(5,5,2)
agent = MCagent(env, 0.8,0.1,20)
agent.train()
