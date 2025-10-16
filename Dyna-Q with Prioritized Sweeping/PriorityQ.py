class Queue:
    def __init__(self):
        self.Q = []
        pass
    def enqueue(self,Value):
        self.Q.append(Value)
        self.Q.sort(key=lambda x: x[0])
    def dequeue(self):
        if len(self.Q) == 0:
            return None

        Val = self.Q[len(self.Q)-1]
        self.Q.pop()
        return Val