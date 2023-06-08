import numpy as np

class FullyConnected():

    delta = 0.1
    def __init__(self,input_size,output_size):
        self.input_size=input_size
        self.output_size=output_size
        self.weights = np.random.uniform(low=0, high=1, size=(self.input_size+1, self.output_size))
        # self.delta=0.1


    def forward(self,input_tensor):
        # input_size=input_tensor.shape[0]
        # batch_size=input_tensor.shape[1]

        add_one = np.ones((input_tensor.shape[0], 1))
        self.input_tensor = np.column_stack((input_tensor, add_one))
        self.output_tensor = np.dot(self.input_tensor, self.weights) # according to test is input*weight
        # print('self.output_tensor,self.input_tensor, self.weights',self.output_tensor.shape,
        #       self.input_tensor.shape, self.weights.shape)

        return self.output_tensor

    def backward(self,error_tensor):
        self.error_input=error_tensor
        # print('back输入',self.error_input.shape)
        self.error_tensor=np.dot(self.error_input,self.weights.T)
        self.error_tensor_out = np.delete(self.error_tensor, -1, axis=1)
        self.get_gradient_weights()
        # print('self.delta',self.delta)
        self.weights=self.weights-self.delta*self.gradient

        return self.error_tensor_out


    def get_gradient_weights(self):
        # self.gradient=np.zeros((self.input_size,self.output_size))
        self.gradient=np.zeros_like(self.weights)
        # print('self.input_tensor,self.error_tensor',self.input_tensor.shape,self.error_tensor.shape)
        # ori_input_tensor=np.delete(self.input_tensor,-1,axis=1)
        self.gradient = np.dot(self.input_tensor.T,self.error_input)
        # print('self.input_tensor,error_tensor',self.input_tensor.shape,self.error_tensor.shape)
        # print('self.gradient.shape',self.gradient.shape)
        return self.gradient



