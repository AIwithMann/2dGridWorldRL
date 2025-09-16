import numpy as np
import random as r

class Env2d:
    def __init__(self, rows:int, cols:int, terminationStates:int)->None:
        self.rows = rows
        self.cols = cols
        self.terminationStates = terminationStates

        random_indices = np.random.choice(rows * cols, self.terminationStates, replace=False)
        self.Rgrid = np.full(shape=(rows,cols),fill_value=-1) #reward grid
        self.Rgrid.flat[random_indices] = 10
        self.terminationStates = np.where(self.Rgrid==10) #termination state
        self.terminationStates = np.array([i for i in self.terminationStates])
        self.terminationStates = self.terminationStates.T
