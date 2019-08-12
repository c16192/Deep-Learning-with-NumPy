from nn.layers import Linear

class TwoInOneOut(object):
    def __init__(self):
        self.layers = [Linear(2, 10)]
