from Layers import *  # *means from document Layers import all
import numpy as np
from copy import *


class NeuralNetwork(object):

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        # net=NeuralNetwork.NeuralNetwork() 第一次作业里的
        # #fcl_1=FullyConnected.FullyConnected(input_size, categories)  4*3
        # fcl_2=FullyConnected.FullyConnected(categories, categories)  3*3

        # net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1e-4), 第二次作业test文件里传过来的
        #                                   Initializers.UniformRandom(),
        #                                   Initializers.Constant(0.1))

        # net.layers.append(fcl_1)/net.layers.append(ReLU.ReLU())/net.layersappend(fcl_2)
        self.layers = []  # 3 object
        self.loss = []
        self.data_layer = None  # net.data_layer = Helpers.IrisData()
        self.loss_layer = None  # net.loss_layer = SoftMax.SoftMax()

        self.optimizer = deepcopy(optimizer)
        self.weights_initializer = deepcopy(weights_initializer)
        self.bias_initializer = deepcopy(bias_initializer)
        # from NeuralNetworkTest.py 的 class TestNeuralNetwork 中 得来的


        # fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        # net.append_trainable_layer(fcl_1)
    def append_trainable_layer(self, layer):
        # self.input_tensor = copy(self.data_layer.input_tensor)
        # self.label_tensor = copy(self.data_layer.label_tensor)
        # self.input_tensor, self.label_tensor = self.data_layer.forward()
        # FullyConnected---def initialize(self, weights_initializer, bias_initializer):
        layer.initialize(self.weights_initializer, self.bias_initializer)
        layer.optimizer = deepcopy(self.optimizer)
        # layer.set_optimizer(self.optimizer)
        self.layers.append(layer)


    def forward(self):
        # python中方法的输出可以有2个，这个按顺序输出2个
        self.input_tensor, self.label_tensor = self.data_layer.forward()  # data_layer=Helpers.IrisData() 里的 forword
        input_tensor = copy(self.input_tensor)
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)  #
            # print(input_tensor.shape)
            # layers=FullyConnected.FullyConnected(input_size, categories) +
            # ReLU.ReLU()+FullyConnected.FullyConnected(categories, categories)

        # loss_layer = SoftMax.SoftMax()
        # rerurn loss=sum of -log(yi)
        self.loss.append(self.loss_layer.forward(input_tensor, self.label_tensor))
        self.loss_out=self.loss_layer.forward(input_tensor, self.label_tensor)
        return self.loss_out

    def backward(self):
        # python中方法的输出可以有2个，这个按顺序输出2个
        # input_tensor, label_tensor = self.data_layer.forward()  # data_layer=Helpers.IrisData()

        # self.loss_layer 是对象Softmax，对象loss_layer中的变量label_tensor输入给它的backward函数
        # 第一次调用backward时里用最后这层的 input--得到prediction-one hot（laber_tensor)= error
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
            #
            # self.weights.append(layer.weights)
        # print(error_tensor)


        pass

    def train(self, iterations):
        for iter in range(iterations):
            self.forward()
            self.backward()
        pass

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        prediction = self.loss_layer.predict(input_tensor)
        return prediction

