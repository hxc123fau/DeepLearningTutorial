import numpy as np
# from Optimization import *
import math


class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        # 依次为 (2, 4, 7)=self.input_shape, (2, 2), (2, 2)
        # self.input_image_shape = input_image_shape
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        # self.recover_tensor=np.zeros()   #用于backward 恢复max pool

    def forward(self, input_tensor):  # input_tensor 2*(2*4*7)=2*56 random
        # print('input_tensor',input_tensor)
        self.batch_size = input_tensor.shape[0]
        self.input_image_shape = input_tensor.shape[1:]
        self.newsize = (
            self.batch_size, self.input_image_shape[0], self.input_image_shape[1], self.input_image_shape[2])
        self.input_tensor = input_tensor.reshape(self.newsize)
        self.recover_tensor = np.zeros(self.newsize)

        ty = 0
        tx = 0
        for j in range(0, self.input_image_shape[1], self.stride_shape[0]):
            if j + self.pooling_shape[0] > self.input_image_shape[1]:
                break
            ty = ty + 1  # 其实可以简化为 ty<=(y-pool)//s
        for k in range(0, self.input_image_shape[2], self.stride_shape[1]):
            if k + self.pooling_shape[1] > self.input_image_shape[2]:
                break
            tx = tx + 1

        output_tensor = np.zeros((self.batch_size, self.input_image_shape[0], ty, tx))
        # print('output_tensor.shape', output_tensor.shape)
        max_position_size = output_tensor.shape
        max_position_size = list(max_position_size)
        max_position_size.insert(4, 2)  # 储存pool窗口提取的 x y 的坐标
        self.max_position_size = tuple(max_position_size)
        # print('self.max_position_size', self.max_position_size)
        self.max_position = np.zeros(self.max_position_size)

        # print(output_tensor.shape)
        for b in range(self.batch_size):
            for i in range(self.input_image_shape[0]):  # z
                newy = 0
                for j in range(0, self.input_image_shape[1], self.stride_shape[0]):  # y -- y_pool
                    # arrive to boundary and not enough area break
                    if j + self.pooling_shape[0] > self.input_image_shape[1]:
                        break
                    newx = 0
                    for k in range(0, self.input_image_shape[2], self.stride_shape[1]):  # x -- x_pool
                        # arrive to boundary and not enough area break
                        if k + self.pooling_shape[1] > self.input_image_shape[2]:
                            break
                        # 提取出pool的窗口
                        pool_b = self.input_tensor[b, i, j:j + self.pooling_shape[0], k:k + self.pooling_shape[1]]
                        # print(newy,newx)
                        output_tensor[b, i, newy, newx] = np.max(pool_b)  # output_tensor新的降采样后的
                        order = np.argmax(pool_b, axis=None)
                        row = order // self.pooling_shape[0]   # 计算在pooling窗口中的位置 取商 得到的是在哪一行
                        column = order % self.pooling_shape[0]   # 取余数 得到在哪一列的位置
                        # max value remain
                        # print('test',j , row, k , column)
                        self.recover_tensor[b, i, j + row, k + column] = self.input_tensor[b, i, j + row, k + column]
                        # print('第几个', b, i, j, k)
                        self.max_position[b, i, newy, newx, 0] = j + row  # y
                        self.max_position[b, i, newy, newx, 1] = k + column  # x
                        newx += 1
                    newy += 1

        # print('newy,newx', newy, newx)
        # print('output_tensor',output_tensor)
        # np.save('max_position',self.max_position)
        self.downsample_size = output_tensor.shape
        # back_size = np.prod((output_tensor.shape[1], output_tensor.shape[2], output_tensor.shape[3]))
        # pool_output_tensor = output_tensor.reshape(self.batch_size, back_size)
        return output_tensor  # last result need to reshape

    def backward(self, error_tensor):
        # self.error_tensor=self.recover_tensor  #上面forward做好的
        # print('error_tensor',error_tensor.shape)
        self.error_tensor = error_tensor
        self.error_tensor=self.error_tensor.reshape(self.downsample_size)  # self.downsample_size 上面forward算好了
        # print('self.error_tensor',self.error_tensor)

        upsample_size=self.newsize
        # print('upsample_size',upsample_size)
        self.pool_upsample = np.zeros(upsample_size)
        # print('self.pool_upsample',self.pool_upsample.shape)


        # print('max_position_size',self.max_position_size)
        # print('self.max_position',self.max_position.shape)
        for b in range(self.batch_size):
            for i in range(self.input_image_shape[0]):
                for j in range(self.max_position_size[2]):
                    for k in range(self.max_position_size[3]):
                        ori_y = self.max_position[b, i, j, k, 0]  # 重叠的相邻窗口可能占据同一个位置的max值
                        ori_x = self.max_position[b, i, j, k, 1]
                        # print('ori_x,ori_y',int(ori_x),int(ori_y))
                        # print('checkyy',self.max_position[b, i, j, k, 0],ori_y)
                        # print('checkxx', self.max_position[b, i, j, k, 1], ori_x)
                        y = int(ori_y)
                        x = int(ori_x)
                        # print('self.pool_upsample', self.pool_upsample.s
                        # hape)
                        # print('input_image_shape', self.input_image_shape)
                        self.pool_upsample[b, i, y, x] += self.error_tensor[b, i, j, k]  # 要叠加

        # print('max_position',self.max_position)
        # np.save('self.error_tensor',self.error_tensor)
        # np.save("pool_up", self.pool_upsample)
        # print('self.pool_upsample',self.pool_upsample)
        # self.pool_upsample = self.pool_upsample.reshape(self.batch_size,
        #                                                upsample_size[1] * upsample_size[2] * upsample_size[3])
        # print('self.pool_upsample',self.pool_upsample)

        return self.pool_upsample
