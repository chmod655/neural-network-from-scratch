import random
import time
import math
from loguru import logger
import numpy

# Class matrix
class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        # create multidimension array fill with zeros
        self.data = [[0] * cols for _ in range(rows)]
        
    # Methods
    @staticmethod
    def transpose(A):
        matrix = Matrix(A.cols, A.rows)
        matrix.map(lambda n, i, j: A.data[j][i])
        return matrix

    '''
        it's nothing less than a map but made by me to do exactly the same 
        thing as the normal map, don't do that use the map that already comes in python
    '''
    def map(self, function):
        self.data = [
            [function(element, i, j) for j, element in enumerate(arr)]
            for i, arr in enumerate(self.data)
        ]
    
    '''
        Generates random numbers and adds to the data, 
        when any variable instantiates the class and using this method will generate a random number for it
    '''
    def randomize(self):
        self.map(lambda elm, i, j: random.random() * 2 - 1)

    
    @staticmethod
    def Map(A, function):
        matrix = Matrix(A.rows, A.cols)
        matrix.data = [
            [function(element, i, j) for j, element in enumerate(arr)]
            for i, arr in enumerate(A.data)
        ]

        return matrix

    '''
        This method will multiply a node with the weight
    '''
    @staticmethod
    def multiply(A, B):
        matrix = Matrix(A.rows, B.cols)
        matrix.map(lambda elm, i, j: sum([A.data[i][k] * B.data[k][j] for k in range(A.cols)]))
        
        
        return matrix

    @staticmethod
    def scalar_multiply(A, scalar):
        matrix = Matrix(A.rows, A.cols)
        matrix.map(lambda elm, i, j: A.data[i][j] * scalar)
        return matrix 
    
    @staticmethod
    def hadamard(A, B):
        matrix = Matrix(A.rows, A.cols)
        matrix.map(lambda elm, i, j: A.data[i][j] * B.data[i][j])
        return matrix 

    @staticmethod
    def subtract(A, B):
        matrix = Matrix(A.rows, A.cols)
        matrix.map(lambda elm, i, j: A.data[i][j] - B.data[i][j])
        return matrix 

    '''
        This method will calculate the weight calculation with no[wi*in+wi1*in1]
    '''
    @staticmethod
    def add(A, B):
        matrix = Matrix(A.rows, A.cols)
        matrix.map(lambda elm, i, j: A.data[i][j] + B.data[i][j])
        return matrix 


    '''
        This method will process the input data to an array
    '''
    @staticmethod
    def array_to_matrix(arr):
        matrix = Matrix(len(arr), 1)
        matrix.map(lambda elm, i, j: arr[i])

        return matrix
    
    @staticmethod
    def matrix_to_array(obj):
        arr = []
        obj.map(lambda elm, i, j: arr.append(elm))

        print(arr)
        return arr

    # Function debug
        # Debug only debug application
    def debug(self, message_print="Function work\n", show_data=0, timeout=0):
        if show_data == 1:
            time.sleep(timeout)
            print(message_print, numpy.array(self.data))
        else:
            time.sleep(timeout)
            logger.debug(message_print)