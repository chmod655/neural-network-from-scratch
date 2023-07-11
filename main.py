import random
import os
from neural_network import NeuralNetwork

if __name__ == '__main__':

    # Dataset
    '''
        Note: This here will be the data to be swallowed by the network
    '''
    dataset = {
        'inputs': [[1, 1], [1, 0], [0, 1], [0, 0]],
        'outputs': [[0], [1], [1], [0]] 
    }


    train = True
    ntk = NeuralNetwork(2,3,1)

    def strain():
        global train
        if train:
            for i in range(10000):
                index = random.randint(0, 3)
                ntk.train(dataset['inputs'][index], dataset['outputs'][index])

            if ntk.predict([0, 0])[0] < 0.04 and ntk.predict([1, 0])[0] > 0.98:
                os.system('clear')
                train = False
                print('Train termined!')


    while True:
        string = f'''
           = [55] - Exit
           x [1] - Prompt
           = [2] - Train NN
           x {'='*20}
        '''
        print(string)
        user = int(input('Prompt: '))
        if user == 1:
            user = int(input('INSERT[0, 1]: '))
            if user == 1:
                print('OUTPUT[1]: ', ntk.predict([1,0]))  
            elif user == 0: print('OUTPUT[0]: ', ntk.predict([0,0]))            
        if user == 2:
            steps = int(input('TRAIN STEPS[10, 100]: '))
            while steps:
                strain()
                train = True
                steps -= 1

        if user == 55: break