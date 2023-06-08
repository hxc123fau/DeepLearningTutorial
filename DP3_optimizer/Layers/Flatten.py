import numpy as np
from scipy import signal

class Flatten():

    def __init__(self):
        pass

    def forward(self,input_tensor):
        for_input_tensor=input_tensor
        self.input_shape=list(for_input_tensor.shape)
        del self.input_shape[0]
        self.batch_size=for_input_tensor.shape[0]
        # output_tensor=np.zeros(batch_size,np.prod(input_shape))

        output_tensor = for_input_tensor.reshape(self.batch_size,np.prod(self.input_shape))

        return output_tensor

    def backward(self,input_tensor):
        back_input_tensor=input_tensor
        recover_shape=self.input_shape
        recover_shape.insert(0,self.batch_size)
        # print('recover_shape',recover_shape)
        # print('self.input_shape',self.input_shape)
        back_output_tensor=back_input_tensor.reshape(recover_shape)
        # print('back_output_tensor',back_output_tensor.shape)

        return back_output_tensor
