class Queue:
    def __init__(self):
        self.Q = []
        pass
    def enqueue(self,Value):
        self.Q.append(Value)
        self.Q.sort(key=lambda x: x[0])
    def dequeue(self):
        Val = self.Q[-1]
        self.Q.pop()
        return val