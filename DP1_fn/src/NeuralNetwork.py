import numpy as np
from Layers import *


class NeuralNetwork():

    def __init__(self):
        self.loss=[]
        self.layers=[]
        self.data_layer=None
        self.loss_layer=None    # net.loss_layer = SoftMax.SoftMax()

    def forward(self):

        input_tensor, self.label_tensor = self.data_layer.forward()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)  #


        self.loss.append(self.loss_layer.forward(input_tensor, self.label_tensor))
        self.loss_out=self.loss_layer.forward(input_tensor, self.label_tensor)
        return self.loss_out

    def backward(self):

        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self,iterations):
        for i in range(iterations):
            self.forward()

            self.backward()

    def test(self,input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        prediction = self.loss_layer.predict(input_tensor)
        return prediction


