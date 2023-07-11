import random

class Matrix:
    # constructor
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        # Create and fill matrix with zeros
        self.data = [[0] * cols for _ in range(rows)]

    # RNG Methods
    def randomize(self):
        self.map(lambda elm, i, j: random.random() * 2 - 1)
    
    # Iteration Methods
    '''
        Note: This method serves to handle network operations 
        such as additions and operations on matrices.
        
        Obs.: it's exactly a map I believe I will remove.
    '''
    def map(self, function):
        self.data = [
            [function(element, i, j) for j, element in enumerate(arr)]
            for i, arr in enumerate(self.data)
        ]

    @staticmethod
    def Map(A, function):
        matrix = Matrix(A.rows, A.cols)

        matrix.data = [
            [function(element, i, j) for j, element in enumerate(arr)] 
            for i, arr in enumerate(A.data)
        ]

        return matrix

    # Conversion Methods
    '''
        Note: This method here as it says converts an array to a matrix, 
        as my neural network is for classification and I use xor problems
        I use inputs like 0's and 1's; So I'm just converting an array to matrix
            Example: 
                input[1, 2]->out:[[1],[2]]
        
        obs.: In short, I'm just converting to a matrix with 2 lines and 1 column
    '''
    @staticmethod
    def array_to_matrix(arr):
        matrix = Matrix(len(arr), 1)
        matrix.map(lambda elm, i, j: arr[i])

        return matrix
    
    '''
        Note: This method converts a matrix to an array, just because it is similar 
        to the method above so not because this explanation would be unnecessary
    '''
    @staticmethod
    def matrix_to_array(obj):
        arr = []
        obj.map(lambda elm, i, j: arr.append(elm))

        return arr

    # Transpose
    '''
        Note: As the name describes this method will transpose the matrix
    '''
    @staticmethod
    def transpose(A):
        matrix = Matrix(A.cols, A.rows)
        matrix.map(lambda n, i, j: A.data[j][i])
        return matrix


    # Operation Methods
    '''
        Note: This method will be used to add the matrices.
        Such as summing the bias in network calculations
    '''
    @staticmethod
    def add(A, B):
        matrix = Matrix(A.rows, A.cols)
        matrix.map(lambda elm, i, j: A.data[i][j] + B.data[i][j])
        return matrix 

    '''
        Note: This method will subtract 2 matrices. 
        Obs.: this method here will be more used in backpropagation (it will literally only be used for backpropagation calculations)
    '''
    @staticmethod
    def subtract(A, B):
        matrix = Matrix(A.rows, A.cols)
        matrix.map(lambda elm, i, j: A.data[i][j] - B.data[i][j])
        return matrix 
    
    '''
        Note: The multiplication method will be used to multiply the matrices
        Obs.: Here it will be used in the calculation with the weight and the knot
    '''
    @staticmethod
    def multiply(A, B):
        matrix = Matrix(A.rows, B.cols)
        matrix.map(lambda elm, i, j: sum([A.data[i][k] * B.data[k][j] for k in range(A.cols)]))
        
        return matrix
    
    '''
        Note: The method will be used to do a scalar multiplication on matrices
    '''
    @staticmethod
    def scalar_multiply(A, scalar):
        matrix = Matrix(A.rows, A.cols)
        matrix.map(lambda elm, i, j: A.data[i][j] * scalar)
        return matrix 
    
    '''
        Note: And this method will apply the hadamard product
    '''
    @staticmethod
    def hadamard(A, B):
        matrix = Matrix(A.rows, A.cols)
        matrix.map(lambda elm, i, j: A.data[i][j] * B.data[i][j])
        return matrix 
