import numpy as np
import random as r

class Env2d:
    def __init__(self, rows:int, cols:int, terminationStates:int)->None:
        self.rows = rows
        self.cols = cols
        self.terminationStates = terminationStates

        random_indices = np.random.choice(rows * cols, self.terminationStates, replace=False)
        self.Rgrid = np.zeros((rows,cols)) #reward grid
        self.Rgrid.flat[random_indices] = 1
        self.terminationStates = np.where(self.Rgrid==1) #termination state
        
