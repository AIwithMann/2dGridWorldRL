import numpy as np

class ContinuousEnv2d:
    def __init__(self, xInterval: int, yInterval: int, numTerminationStates: int) -> None:
        self.xInterval = xInterval
        self.yInterval = yInterval
        self.n = numTerminationStates

        x = np.random.uniform(low=0, high=self.xInterval, size=self.n)
        y = np.random.uniform(low=0, high=self.yInterval, size=self.n)
        self.terminationStates = np.column_stack((x, y))

    def reward(self, x: float, y: float):
        if any((np.round(x) == np.round(xs)) and (np.round(y) == np.round(ys)) for xs, ys in self.terminationStates): return 1
        else: return 0
hello = ContinuousEnv2d(10, 10, 2)
