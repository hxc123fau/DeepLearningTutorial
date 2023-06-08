import numpy as np
from Layers import Base

class Dropout:
    def __init__(self,probability):
        self.probability=probability
        self.phase = Base.Phase.train

    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        input_size=list(input_tensor.shape)
        # dropout_number=int((1.0-self.probability)*input_size[1])
        # print('dropout_number',dropout_number)
        self.dropout=np.ones_like(input_tensor)
        # print('self.dropout',self.dropout.shape)
        for i in range(input_size[0]):
            for j in range (input_size[1]):
                # self.dropout[i,0:dropout_number]=0
                random_p=np.random.uniform(0, 1)
                if random_p >=self.probability:
                    self.dropout[i,j]=0

        output_tensor=np.multiply(self.dropout, self.input_tensor)   #do dropout
        if self.phase is Base.Phase.train:
            output_tensor=output_tensor/self.probability
        else:  # test
            output_tensor =input_tensor

        print('output_tensor',output_tensor[1])
        return output_tensor


    def backward(self,error_tensor):
        if self.phase is Base.Phase.train:
            error_tensor_out= np.multiply(self.dropout, error_tensor)
        else:
            error_tensor_out=error_tensor

        return error_tensor_out




