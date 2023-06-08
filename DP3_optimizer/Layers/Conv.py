import numpy as np
from scipy import signal
from copy import *
import math
import cv2
from Optimization import *

# from Layers import *
# from Optimization import *

class Conv:

    weights=None
    # bias=np.array([0])
    bias=None
    optimizer=None

    def __init__(self,  stride_shape, kernel_shape, number_kernels,*parameter):
    # def __init__(self, stride_shape, *parameter):
    #     self.img_shape = input_image_shape
        self.stride_shape = stride_shape
        self.kernel_shape = kernel_shape  #(3, 5, 8)
        self.number_kernels = number_kernels
        # self.kernel_shape=parameter[0]
        # self.number_kernels =parameter[1]

        # in sequence 4 kernel , 3 layer , 10*14
        # self.weights=conv.weights  #怎么继承过来 ？？？ Test 340 line
        # self.batch_size=2

        # kernel_dimension=self.kernel_shape.ndim
        kernels_shape=list(self.kernel_shape)
        kernels_shape.insert(0,self.number_kernels) #加入了核的个数
        kernels_shape=tuple(kernels_shape)
        print('kernels_shape',kernels_shape)

        # kernels_shape = (self.number_kernels, self.kernel_shape[0], self.kernel_shape[1], self.kernel_shape[2])
        self.weights = np.random.uniform(low=0, high=1, size=kernels_shape)
        self.bias = np.random.uniform(0, 1, (self.number_kernels ))
        # self.bias = np.ones(number_kernels) * 0.5

        # self.biasMat = np.ones(biasmat_size)*self.bias[0]  #(b,ky,(!kx) ) #直接移到forward里做了


        if len(parameter)==1:
            self.delta=parameter[0]

        else:
            self.delta = 1


        # gradient_weight_shape=list(kernels_shape)
        # gradient_weight_shape.insert(0,self.batch_size)
        # gradient_weight_shape=tuple(gradient_weight_shape)
        # self.gradient_weight = np.zeros(gradient_weight_shape)  #(b,k,z,y,x)

    def forward(self, input_tensor):  # input_tensor我想先变reshape成 input(batch,(3, 10, 14)) 再卷积,不然不能直接卷
        self.batch_size = input_tensor.shape[0]
        self.img_shape=input_tensor.shape[1:]
        self.reshape_img_size=list(self.img_shape)
        self.reshape_img_size.insert(0,self.batch_size)
        self.reshape_img_size=tuple(self.reshape_img_size) # 加入了batchsize的
        self.input_tensor = np.array(input_tensor)  # input(batch,(3, 10, 14))  kernel(3, 5, 8)
        self.input_tensor = self.input_tensor.reshape(self.reshape_img_size)  # reshape as (b,z,y,x)

        biasmat_size=list(self.img_shape)
        del biasmat_size[0]
        biasmat_size.insert(0, self.number_kernels)
        biasmat_size.insert(0,self.batch_size)   #加在每个核的最底层 4kernel *10*14
        self.biasmat_size=biasmat_size   # rest size is 2d y,x
        self.biasMat = np.ones(biasmat_size)
        self.biasMat=np.ones(self.biasmat_size) # (b,k,y,x)
        # print('self.biasmat_size',self.biasmat_size)

        for b in range(self.biasmat_size[0]):
            for k in range(self.number_kernels):
                self.biasMat[b,k] = self.biasMat[b,k] * self.bias[k]

        add_input_size=list(self.img_shape)
        del add_input_size[0]
        add_input_size.insert(0,self.batch_size)
        add_input_size=tuple(add_input_size) # (batch,y,x) 一层的
        self.add_input=np.ones(add_input_size)
        # print(self.input_tensor.shape)

        self.output_size=self.reshape_img_size
        self.output_size=list(self.output_size)
        self.output_size[1]=self.number_kernels  # 第二个 number z change to self.number_kernels
        self.output_size=tuple(self.output_size)
        self.output_tensor = np.zeros(self.output_size)

        get_input_size = np.zeros(self.img_shape)  # be used to judge the input dimension

        if not self.weights is None:
            self.weights=self.weights
            # print(self.weights)

        if get_input_size.ndim==3:
            sub_y = self.img_shape[1] // self.stride_shape[0]
            if self.img_shape[1] % self.stride_shape[0] != 0:  # if rest still has space it can still conv see from test file
                sub_y += 1
            sub_x = self.img_shape[2] // self.stride_shape[1]
            if self.img_shape[2] % self.stride_shape[1] != 0:
                sub_x += 1
            self.sub_output_tensor = np.zeros((self.batch_size, self.number_kernels, sub_y, sub_x))
        else:
            sub_y = self.img_shape[1] // self.stride_shape[0]
            if self.img_shape[1] % self.stride_shape[0] != 0:  # if rest still has space it can still conv see from test file
                sub_y += 1
            self.sub_output_tensor = np.zeros((self.batch_size, self.number_kernels, sub_y))

        # print('biasMat',self.biasMat.shape)
        # print('output_tensor',self.output_tensor.shape)
        # print('self.batch_size',self.batch_size)
        for b in range(self.batch_size):  # convolution  first
            for k in range(self.number_kernels):   # one kernel get one layer
                    kernel = self.weights[k]
                    # print('试验',self.input_tensor[b].shape,self.weights.shape)
                    temp = signal.correlate(self.input_tensor[b], kernel, mode='same')
                    self.output_tensor[b, k] = temp[math.floor(self.img_shape[0] / 2)]  # 用floor因为从0开始,提取中间层
                    self.output_tensor[b, k] += self.biasMat[b,k]  #每层每个小格子都加bias

        for b in range(self.batch_size):  # and then downsampling  x,y降采样，z还是3层不变
            for i in range(sub_y):  # y  img:10  kernel:5
                if  get_input_size.ndim==3:
                    for j in range(sub_x):  # x  img:14  kernel:8 先某列，n行提取
                        self.sub_output_tensor[b, :, i, j] = self.output_tensor[b, :, i * self.stride_shape[0],
                                                             j * self.stride_shape[1]]
                else:
                    self.sub_output_tensor[b, :, i] = self.output_tensor[b, :, i * self.stride_shape[0]]

        # print(self.sub_output_tensor.shape)

        sub_out_shape = self.sub_output_tensor.shape  # 卷积+降采样后的原形状先保存 (b,z,y,x)
        sub_out_reshape=list(sub_out_shape)
        del sub_out_reshape[0]
        sub_out_reshape=tuple(sub_out_reshape)
        return_shape = np.prod(sub_out_reshape)  # (z*y*x) or (z,y)
        # output_next_tensor = self.sub_output_tensor.reshape((self.batch_size, return_shape))  # (b,z*y*x)
        # print(self.sub_output_tensor.shape)

        return self.sub_output_tensor

    def backward(self, error_tensor):  # 先升采样再卷积
        self.error_tensor = np.array(error_tensor)
        get_input_size=np.zeros(self.img_shape)
        reshape_size=None
        if get_input_size.ndim==3:  #需要得到之前降采样后的大小，因为进入的是组合过后的无序数组
            sub_y = self.img_shape[1] // self.stride_shape[0]
            if self.img_shape[1] % self.stride_shape[0] != 0:  # if rest still has space it can conv see from test file
                sub_y += 1
            sub_x = self.img_shape[2] // self.stride_shape[1]
            if self.img_shape[2] % self.stride_shape[1] != 0:  #是不是应该改成 余数< kenel_x/2
                sub_x += 1
            reshape_size = (self.batch_size, self.number_kernels, sub_y, sub_x)
            upsamlpe_size=(self.batch_size, self.number_kernels, self.img_shape[1], self.img_shape[2])
            ori_yx_size = (self.img_shape[2], self.img_shape[1])  #此处 x，y不是y，x
        else:
            sub_y = self.img_shape[1] // self.stride_shape[0]
            if self.img_shape[1] % self.stride_shape[0] != 0:  # if rest still has space it can conv see from test file
                sub_y += 1
                reshape_size = (self.batch_size, self.number_kernels, sub_y)
            upsamlpe_size=(self.batch_size,self.number_kernels,self.img_shape[1])  #2,4,15
            ori_zy_size = (self.img_shape[1],self.number_kernels) #剩下b,z,y后 对z,y可以直接resize z=4,y=15在resize写成y,z

        # print(reshape_input_size)
        self.error_tensor = self.error_tensor.reshape(reshape_size)  # (b,z,y,x)
        self.error_upsample_tensor = np.zeros(upsamlpe_size)
        # print(self.error_upsample_tensor.shape)
        # error_con_out_tensor = np.zeros(self.img_shape) #最后的输出与输入一样

          # 降采样前 x y 层的size,升采样要恢复的 opencv的resize顺序换
        # print(ori_yx_size,self.error_tensor.shape)
        if get_input_size.ndim == 3:
            for i in range(self.error_tensor.shape[2]):
                for j in range(self.error_tensor.shape[3]):  # （z层）每层都upsampling
                    # print(self.error_tensor[b, z, :, :].shape,ori_yx_size)
                    #     self.error_upsample_tensor[b, z, :, :] = cv2.resize(self.error_tensor[b, z, :, :], ori_yx_size,
                    #                                                         interpolation=cv2.INTER_CUBIC)
                    self.error_upsample_tensor[:, :,i*self.stride_shape[0],j*self.stride_shape[1]]=self.error_tensor[:,:,i,j]
        else:  #z y resize directly
                # print(self.error_tensor.shape,ori_zy_size)
            for i in range(self.error_tensor.shape[2]):
                # self.error_upsample_tensor[:,:, i] = cv2.resize(self.error_tensor[b, :, :], ori_zy_size,
                #                                                         interpolation=cv2.INTER_CUBIC) # 2
                self.error_upsample_tensor[:, :, i*self.stride_shape[0]] =self.error_tensor[:,:,i]

                # new_kernels=np.zeros((self.kernel_shape[0],self.number_kernels,self.kernel_shape[1],self.kernel_shape[2]))
        # 新核重组
        new_kernels_size=list(self.kernel_shape)
        new_kernels_size.insert(1,self.number_kernels)
        new_kernels_size=tuple(new_kernels_size)
        new_kernels=np.zeros(new_kernels_size)
        for i in range(self.kernel_shape[0]): #kernel_shape[0]=3
            for j in range(self.number_kernels):  #self.number_kernels=4
                if get_input_size.ndim == 3:
                    new_kernels[i, j, :, :] = self.weights[j, i, :, :]
                else:
                    new_kernels[i, j, :] = self.weights[j, i, :]

        # print('new_kernels',new_kernels.shape)
        # 输出大小 output_size
        error_out_size=list(self.img_shape)
        error_out_size.insert(0,self.batch_size)
        error_out_size=tuple(error_out_size)
        error_con_out_tensor = np.zeros(error_out_size)  # 最后的输出与输入一样

        # print('error_upsample_tensor',self.error_upsample_tensor.shape)
        # print('error_con_out_tensor',error_con_out_tensor.shape)

        for b in range(self.batch_size):  # 卷积 first 然后 upsampling ,in order to reduce computing？
            for i in range(new_kernels.shape[0]): # 新核的数量 new_kernels.shape[0]=3
                kernel = new_kernels[i]  # 每次赋给的kernel 4*5*8
                #下方的卷积还缺 2b,4层的biasMat和kernel的卷积，卷积回去结果类似input底层加的全1，因为单层的test里还不需要
                kernel = kernel[::-1]  #就做z方向的翻转，这样就相当于convolve的时候就y，x在convolve
                # print('kernel',kernel.shape)
                extract=math.floor(self.error_upsample_tensor.shape[1] / 2)
                if get_input_size.ndim == 3:
                    error_con_out_tensor[b, i, :, :] = signal.convolve(self.error_upsample_tensor[b, :, :, :], kernel,
                                                                       mode='same')[extract,:, :]
                else:
                    error_con_out_tensor[b, i, :] = signal.convolve(self.error_upsample_tensor[b, :, :], kernel,
                                                                       mode='same')[extract, :]
                    # print(error_con_out_tensor[b, i, :].shape)
        # print(error_con_out_tensor.shape)

        if not self.optimizer is None:
            # self.set_optimizer(self, optimizer) #外部已经调用过了,backward前给self.optimizer写好了，这里不需要了
            self.get_gradient_weights()
            self.get_gradient_bias()
            self.weights = self.optimizer.calculate_update(1, self.weights, self.gradient_weight_out)
            self.bias = self.optimizer.calculate_update(1, self.bias,  self.gradient_bias)


        error_out_tensor = error_con_out_tensor.reshape(self.batch_size,np.prod(self.img_shape))
        return error_con_out_tensor  # (batch,z,y,x)

    def set_optimizer(self, optimizer):
        # opti = Optimizers.Sgd(global_delta)
        # self.weights = opti.calculate_update(individual_delta, weight_tensor,gradient_tensor)  #
        self.optimizer=deepcopy(optimizer)
        # self.get_gradient_weight()
        # self.weights = optimizer.calculate_update(1, self.weights, self.gradient_weight_out)

    def initialize(self, weight_initializer, bias_initializer):
        fan_in=self.weights.shape[1] * self.weights.shape[2] * self.weights.shape[3]
        fan_out=self.number_kernels * self.weights.shape[2] * self.weights.shape[3]
        self.weights = weight_initializer.initialize(self.weights,fan_in,fan_out) #use TestInitializer.initialize
        self.bias = bias_initializer.initialize(self.bias,None,None)  # 输入一个与kernel y，z大小一样的2d矩阵

        # self.weights = weight_initializer.initialize(self.weights.shape,self.weights.shape[1],self.weights.shape[0])
        # self.bias = bias_initializer.initialize(self.bias.shape,self.bias.shape[1],self.bias.shape[0])
        return self.weights, self.bias

    def get_gradient_weights(self):
        # self.valid_con_output如果不行，从进来的error_tensor中提取中心valid那一部分也可以  学妹说这里现在用padding
        get_input_size=np.zeros(self.img_shape)
        if get_input_size.ndim==3:
            ky = self.kernel_shape[1]
            kx = self.kernel_shape[2]
            pad_u = math.ceil((ky - 1) / 2.0)  # python默认左上方向补零 所以用ceil能得到vaild值
            pad_d = math.floor((ky - 1) / 2.0)
            pad_l = math.ceil((kx - 1) / 2.0)  # python默认左上方向补零 所以用ceil能得到vaild值
            pad_r = math.floor((kx - 1) / 2.0)
            self.pad_input_tensor = np.zeros(
                (self.batch_size, self.img_shape[0], self.img_shape[1] + ky - 1, self.img_shape[2] + kx - 1))
            self.pad_input_tensor[:, :, pad_u:-pad_d, pad_l:-pad_r] = self.input_tensor  # 补完零的
            # print('self.pad_input_tensor', self.pad_input_tensor.shape)
        else:
            ky = self.kernel_shape[1]
            pad_u = math.ceil((ky - 1) / 2.0)  # python默认左上方向补零 所以用ceil能得到vaild值
            pad_d = math.floor((ky - 1) / 2.0)
            self.pad_input_tensor = np.zeros(
                (self.batch_size, self.img_shape[0], self.img_shape[1] + ky - 1))
            self.pad_input_tensor[:, :, pad_u:-pad_d] = self.input_tensor  # 补完零的


        # #卷出weight的形状给出去
        gradient_weight_out_size=list(self.kernel_shape)
        gradient_weight_out_size.insert(0,self.number_kernels)
        gradient_weight_out_size=tuple(gradient_weight_out_size)  #(b,z,y,x)
        self.gradient_weight_out=np.zeros(gradient_weight_out_size)
        temp_weights=np.zeros_like(self.gradient_weight_out)
        # print('gradient_weight_out_size',self.gradient_weight_out.shape)

        # self.error_upsample_tensor[b, i, :, :]

        # self.error_upsample_tensor = np.rot90(self.error_upsample_tensor, 2, (2, 3))
        for b in range(self.batch_size):
            for i in range(self.number_kernels):  #error有4层，每次输入一层，最后get4个gradient——weight
                if get_input_size.ndim == 3:
                    temp = signal.correlate(self.pad_input_tensor[b, :, :, :],
                                           np.expand_dims(self.error_upsample_tensor[b, i, :, :], axis=0),
                                           mode='valid')

                    temp_weights[i]=temp

                else:
                    temp = signal.correlate(self.pad_input_tensor[b, :, :],
                                           np.expand_dims(self.error_upsample_tensor[b, i, :], axis=0),
                                           mode='valid')
                    temp_weights[i] = temp

            self.gradient_weight_out=self.gradient_weight_out+ temp_weights

        return self.gradient_weight_out

    def get_gradient_bias(self):
        self.gradient_bias = np.zeros_like(self.bias)
        for i in range(self.number_kernels):
            for b in range(self.error_tensor.shape[0]):
                self.gradient_bias[i] += np.sum(self.error_tensor[b, i])
        return self.gradient_bias
