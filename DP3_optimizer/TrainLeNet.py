from Layers import Helpers
# from Models.LeNet import LeNet
import NeuralNetwork
import matplotlib.pyplot as plt
import os.path


import sys
#sys.path.append('C:/Users/harry/Desktop/dp3_src/Models/LeNet.py')
from Models import LeNet
# from Models.LeNet import build


batch_size = 50
mnist = Helpers.MNISTData(batch_size)
mnist.show_random_training_image()
# print('lenet000')
if os.path.isfile('trained/LeNet'):
    # print('lenet111')
    net = NeuralNetwork.load('trained/LeNet', mnist)
    print('检查net',net)
else:
    print('222')
    # print('mnist',mnist.forward())
    LeNet_class=LeNet.LeNet()
    LeNet_class.data_layer = mnist
    net = LeNet_class.build()
    # LeNet.data_layer = mnist
    # net = build()
    net.data_layer = mnist
    print('net00',LeNet_class.build())

print('333')
print('net11',net)
# net.train(300)
net.train(10)

# print('net22',net.layers)
# a = {'hello': 'world'}
# NeuralNetwork.save('tt.pickle',net)
# NeuralNetwork.save('trained/LeNet', net)
NeuralNetwork.save('trained/LeNet', net)

print('net.loss',type(net.loss))
plt.figure('Loss function for training LeNet on the MNIST dataset')
plt.plot(net.loss, '-x')
plt.show()

data, labels = net.data_layer.get_test_set() # return self.test, self.testLabels

results = net.test(data)

accuracy = Helpers.calculate_accuracy(results, labels)
print('\nOn the MNIST dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')


