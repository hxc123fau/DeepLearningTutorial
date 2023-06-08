import numpy as np
from copy import *


class FullyConnected:

    input_tensor = np.empty((0, 0))
    output_tensor = np.empty((0, 0))
    error_tensor = np.empty((0, 0))
    gradient = np.empty((0, 0))
    error = np.empty((0, 0))
    # delta = 0
    optimizer = None

    def __init__(self,input_size,output_size,*delta):
        self.input_size=input_size
        self.output_size=output_size
        self.weights = np.random.uniform(low=0, high=1, size=(self.input_size+1, self.output_size))
        # print('fully_weights',self.weights.shape)
        if len(delta)==1:
            self.delta=delta
        else:
            # self.delta=0.1
            self.delta = 5e-1

    def get_weights(self):
        # print('ful_self.weights',self.weights.shape)
        return self.weights


    def set_weights(self, weights):
        self.weights = weights

    def initialize(self, weights_initializer, bias_initializer):
        # print('self.weights[:-1, :]',self.weights[:-1, :].ndim)
        weights = weights_initializer.initialize(self.weights[:-1, :],None,None)
        bias = np.expand_dims(self.weights[-1, :], axis=0)
        bias = bias_initializer.initialize(bias,None,None)
        # print('biasout',self.bias)
        self.weights = np.concatenate((weights, bias), axis=0)

    def forward(self, input_tensor):
        add_one = np.ones((input_tensor.shape[0], 1))   # (batch_size,1)
        # self.input_tensor = np.column_stack((input_tensor, add_one))
        # print('add_one',add_one.shape)
        self.input_tensor=np.hstack((input_tensor, add_one))
        # print('self.input_tensor',self.input_tensor.shape)
        # print('self.weights',self.weights.shape)
        self.output_tensor = np.dot(self.input_tensor, self.weights)  # according to test is input*weight

        return self.output_tensor

    def backward(self, error_tensor):
        self.error_input=error_tensor
        # print('back输入',self.error_input.shape)
        self.error_tensor=np.dot(self.error_input,self.weights.T)
        # print('self.error_tensor',self.error_tensor.shape,self.error_input.shape,(self.weights.T).shape)
        self.error_tensor_out = np.delete(self.error_tensor, -1, axis=1)
        self.get_gradient_weights()
        if (self.optimizer):
            self.weights = self.optimizer.calculate_update(self.delta, self.weights, self.gradient)
        # print('self.delta',self.delta)
        # self.weights=self.weights-self.delta*self.gradient

        return self.error_tensor_out

    def get_gradient_weights(self):
        self.gradient=np.zeros_like(self.weights)
        self.gradient = np.dot(self.input_tensor.T,self.error_input)
        return self.gradient

    def set_optimizer(self, optimizer):
        self.optimizer = deepcopy(optimizer)

