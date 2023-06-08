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
        # print('fully_self.weights ',self.weights.shape)
        if len(delta)==1:
            self.delta=delta
        else:
            # self.delta=3e-1
            self.delta = 5e-1


    def initialize(self, weights_initializer, bias_initializer):
        # print('self.weights[:-1, :]',self.weights[:-1, :].ndim)
        # print('initi_self.weights.shape',self.weights.shape)
        # weights = weights_initializer.initialize(self.weights[:-1, :],None,None)
        weights = weights_initializer.initialize(self.weights.shape, self.weights.shape[0] - 1, self.weights.shape[1])
        bias = np.expand_dims(self.weights[-1, :], axis=0)
        bias = bias_initializer.initialize(bias,None,None)
        # print('biasout',self.bias)
        self.weights = np.concatenate((weights, bias), axis=0)
        # print('forward_self.weights11', self.weights.shape)

    def forward(self, input_tensor):
        # print('input_tensor',input_tensor.shape)
        # print('forward_self.weights22',self.weights.shape)
        # if len(input_tensor.shape)!=2:
        #     new_shape=(input_tensor.shape[0],np.prod(input_tensor.shape[1:]) )
        #     print('new_shape',new_shape)
        #     input_tensor=input_tensor.reshape(new_shape)

        add_one = np.ones((input_tensor.shape[0], 1))
        self.input_tensor = np.column_stack((input_tensor, add_one))
        # print('forward_self.weights33',self.weights.shape)
        self.output_tensor = np.dot(self.input_tensor, self.weights)  # according to test is input*weight

        return self.output_tensor

    def backward(self, error_tensor):
        self.error_input=error_tensor
        # print('back输入',self.error_input.shape)
        self.error_tensor=np.dot(self.error_input,self.weights.T)
        self.error_tensor_out = np.delete(self.error_tensor, -1, axis=1)
        self.get_gradient_weights()
        if (self.optimizer):
            self.weights = self.optimizer.calculate_update(self.delta, self.weights, self.gradient)
        else:
            # self.weights = self.weights - self.delta * self.gradient
            self.weights=self.weights
        # print('self.delta',self.delta)
        # self.weights=self.weights-self.delta*self.gradient

        return self.error_tensor_out

    def get_gradient_weights(self):
        self.gradient=np.zeros_like(self.weights)
        self.gradient = np.dot(self.input_tensor.T,self.error_input)
        return self.gradient

    def set_optimizer(self, optimizer):
        self.optimizer = deepcopy(optimizer)

