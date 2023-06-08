import numpy as np
import unittest

# input_tensor = np.arange(36).reshape((9,4))
#
# tt=np.max(input_tensor)
# res=np.subtract(input_tensor,tt)
# exp_x=np.exp(res)
#
# pred=np.divide(exp_x, np.expand_dims(np.sum(exp_x, axis=1),1))
# # print('input_tensor',input_tensor)
# # print('tt',tt)
# # print('res',res)
# # print('exp_x',exp_x)
# print(pred.shape)

class TestNeuralNetwork(unittest.TestCase):

    def test_data_access(self):
        out=np.arange(9).reshape(3,3)
        out2 = np.linspace(5,10,9).reshape(3, 3)

        print(out,out2)


        self.assertNotEqual(1, 2)