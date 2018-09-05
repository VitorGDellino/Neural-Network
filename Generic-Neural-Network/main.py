from mlp import Mlp

def main():
    nn = Mlp(2, 2, 1)
    nn.feed_forward([1,0])
    nn.show()


if __name__ == "__main__":
    main()