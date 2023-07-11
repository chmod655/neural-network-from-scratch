import math
from matrix import Matrix

# Activation Functions
'''
    Note: This sigmoid function takes as input any real value and outputs values ​​in the range 0 to 1.
'''
def sigmoid(x, i, j):
    return 1 / (1 + math.exp(-x))

'''
    Note: This function derives the sigmoid function
'''
def dsigmoid(x, i, j):
    return x * (1 - x)


# Neural network class
class NeuralNetwork:
    '''
        Node: When instantiated the constructor will have layers with such numbers of nodes. 
        This constructor initializes with random numbers for neural network.
        
        Obs.: Soon I will implement a "memory"
    '''
    def __init__(self, i_nodes, h_nodes, o_nodes) -> None:
        # Quantity nodes(neurons- i, h, o)
        self.i_nodes = i_nodes
        self.h_nodes = h_nodes
        self.o_nodes = o_nodes

        # Bias [Input To hide]
        self.bias_ih = Matrix(self.h_nodes, 1)
        self.bias_ih.randomize()

        # Bias [Hide To Output]
        self.bias_ho = Matrix(self.o_nodes, 1)
        self.bias_ho.randomize()

        # Weight [Input to hide]
        self.weights_ih = Matrix(self.h_nodes, self.i_nodes)
        self.weights_ih.randomize()

        # Weight [Hide to output]
        self.weights_ho = Matrix(self.o_nodes, self.h_nodes)
        self.weights_ho.randomize()
        
        # Learning rate
        self.learning_rate = 0.1

    # Train Method
    '''
        Note: This method will train the neural network and after reaching the expected value it stops.
        in this method the backpropagation is working
    '''
    def train(self, arr, target):                                                                                                    
        # INPUT -> HIDDEN
        input = Matrix.array_to_matrix(arr)
        hidden = Matrix.multiply(self.weights_ih, input)        
        hidden = Matrix.add(hidden, self.bias_ih)
        hidden.map(sigmoid)

        # HIDDEN -> OUTPUT
        output = Matrix.multiply(self.weights_ho, hidden)
        output.add(output, self.bias_ho)
        output.map(sigmoid)

        # BACKPROPAGATION
        # OUTPUT -> HIDDEN
        expected = Matrix.array_to_matrix(target)
        output_error = Matrix.subtract(expected, output)
        d_output = Matrix.Map(output, dsigmoid)

        hidden_t = Matrix.transpose(hidden)

        gradient = Matrix.hadamard(d_output, output_error)
        gradient = Matrix.scalar_multiply(gradient, self.learning_rate)

        # Ajust bias_ho
        self.bias_ho = Matrix.add(self.bias_ho, gradient)

        weight_ho_delta = Matrix.multiply(gradient, hidden_t)
        self.weights_ho = Matrix.add(self.weights_ho, weight_ho_delta)
       
        # HIDDEN -> INPUT
        weight_ho_t = Matrix.transpose(self.weights_ho)
        hidden_error = Matrix.multiply(weight_ho_t, output_error)
        d_hidden = Matrix.Map(hidden, dsigmoid)
        
        input_t = Matrix.transpose(input)

        gradient_h = Matrix.hadamard(hidden_error, d_hidden)
        gradient_h = Matrix.scalar_multiply(gradient_h, self.learning_rate)

        
        # Ajust bias_ho
        self.bias_ih = Matrix.add(self.bias_ih, gradient_h)
    
        weight_ih_delta = Matrix.multiply(gradient_h, input_t)
        self.weights_ih = Matrix.add(self.weights_ih, weight_ih_delta)

    # Predicit Method
    '''
        Note: After training the network will return a value to the prompt, predict will do the classification process after training.

        Obs.: After implementing data persistence, as soon as the results are achieved, predict will load the values
    '''
    def predict(self, arr):
        # INPUT -> HIDDEN
        input = Matrix.array_to_matrix(arr)
        hidden = Matrix.multiply(self.weights_ih, input)        
        hidden = Matrix.add(hidden, self.bias_ih)
        hidden.map(sigmoid)

        # HIDDEN -> OUTPUT
        output = Matrix.multiply(self.weights_ho, hidden)
        output.add(output, self.bias_ho)
        output.map(sigmoid)

        output = Matrix.matrix_to_array(output)
        
        # OUTPUT
        return output