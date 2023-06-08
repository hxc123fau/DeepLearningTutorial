from Layers import *  # *means from document Layers import all
import numpy as np
from copy import *
import pickle
from Layers import Base
from Optimization import *
import os


class NeuralNetwork():

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
        self.phase=None
        self.regularization_loss=0
        # self.input_tensor=None
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
        # input_tensor, label_tensor = self.data_layer.forward()  # data_layer=Helpers.IrisData() 里的 forword
        # np.savetxt("t_filename.txt", self.layers)
        # self.loss = []
        self.input_tensor, self.label_tensor = self.data_layer.forward()
        input_tensor = copy(self.input_tensor)
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)  #
            # print(input_tensor.shape)
            # layers=FullyConnected.FullyConnected(input_size, categories) +
            # ReLU.ReLU()+FullyConnected.FullyConnected(categories, categories)

        # loss_layer = SoftMax.SoftMax()
        # self.loss.append(self.loss_layer.forward(input_tensor, self.label_tensor))  # rerurn loss=sum of -log(yi)
        # print('self.layers',self.layers)
            if isinstance(layer, FullyConnected.FullyConnected or Conv.Conv):
                if self.optimizer is not None :
                    base_class=Base.Base_class()
                    base_class.regularizer=Constraints.L2_Regularizer(4e-4)
                    self.regularization_loss+=base_class.calculate_regularization_loss(layer)
                    print('regularization_loss',self.regularization_loss)

        self.loss_out=self.loss_layer.forward(input_tensor, self.label_tensor)+self.regularization_loss
        # print('self.loss_out',type(self.loss_out),type(self.loss))
        self.loss.append(self.loss_out)  # rerurn loss=sum of -log(yi)
        print('self.loss结果',self.loss,self.loss_out)
        return self.loss_out

    def backward(self):
        # python中方法的输出可以有2个，这个按顺序输出2个  label_tensor is random value??
        # input_tensor, label_tensor = self.data_layer.forward()  # data_layer=Helpers.IrisData()

        # self.loss_layer 是对象Softmax，对象loss_layer中的变量label_tensor输入给它的backward函数
        # 第一次调用backward时里用最后这层的 input--得到prediction-one hot（laber_tensor)= error
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

        pass

    def train(self, iterations):
        # self.loss=[]
        for iter in range(iterations):
            self.forward()
            self.backward()
        pass

    def test(self, input_tensor):
        self.set_phase(Base.Phase.test)
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        prediction = self.loss_layer.predict(input_tensor)
        return prediction

    def set_phase(self,phase):
        for layer in self.layers:
            # if (layer == Dropout):
            # print('layer输出',layer)
            if isinstance(layer,Dropout.Dropout):
                layer.phase=phase
                # print('Layers.Dropout.Dropout')
            if isinstance(layer, BatchNormalization.BatchNormalization):
                layer.phase = phase

    def del_data_layer(self):
        self.data_layer=None

    def set_data_layer(self,data_layer):
        self.data_layer=data_layer


# def save(filename,obj):
#     with open(filename, 'wb') as fn:
#         pickle.dump(obj, fn)      # 将obj 写入fn（也就是filename）中


# def load(filename,data_layer):
#     with open(filename, 'rb') as fn2:
#         net = pickle.load(fn2)
#     net.data_layer = data_layer

# def load(filename, data_layer):
#     net = pickle.load(filename)
#     net.data_layer = data_layer
#     return net


def save(filename, net):
    # temp=net.data_layer
    # f = open(filename, 'wb')
    # net.del_data_layer()
    # # net.__setstate__()
    # pickle.dump(net, f)
    # net.data_layer=temp
    # f.close()

    dir, filename = os.path.split(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)

    temp = net.data_layer
    with open(filename, "wb") as f:
        net.del_data_layer()
        pickle.dump(net, f)
        net.data_layer = temp


# def save(filename,net):
#     pickle.dump(net,open(filename,"wb"))
#     return

def load(filename, data_layer):
    # f = open(filename, 'rb')
    # net = pickle.load(f)
    # net.set_data_layer(data_layer)
    # # net.__getstate__(data_layer)
    # f.close()
    # return net

    net = pickle.load(open(filename, "rb"))
    net.data_layer = data_layer
    return net


#
# fn='filename.pickle'
# with open(fn, 'wb') as handle:
#     pickle.dump(addition(), handle)
#
# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

