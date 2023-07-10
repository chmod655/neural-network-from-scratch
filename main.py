import random
from neural_network import NeuralNetwork

if __name__ == '__main__':
    dataset = {
        'inputs': [[1, 1], [1, 0], [0, 1], [0, 0]],
        'outputs': [[0], [1], [1], [0]] 
    }

    train = True
    ntk = NeuralNetwork(2,3,1)

    if train:
        for i in range(1000000):
            index = random.randint(0, 3)
            ntk.train(dataset['inputs'][index], dataset['outputs'][index])

        if ntk.predict([0, 0])[0] < 0.04 and ntk.predict([1, 0])[0] > 0.99:
            train = False
            print('Train termined!')
