from mlp import Mlp
import numpy as np
def main():
    nn = Mlp(2, 2, 1)
    nn.backpropagation([[1,0,1],[0,1,1],[0,0,0],[1,1,0]])
    nn.feed_forward([1,1])
    nn.show()
    nn.feed_forward([1,0])
    nn.show()
    nn.feed_forward([0,1])
    nn.show()
    nn.feed_forward([0,0])
    nn.show()


if __name__ == "__main__":
    main()