import numpy as np
import math
from copy import *


class Sgd:

    def __init__(self, global_delta):
        self.global_delta = global_delta

    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        delta = self.global_delta * individual_delta
        # print('weight_tensor',weight_tensor)
        self.weight_tensor = weight_tensor - delta * gradient_tensor
        # print('self.weight_tensor',self.weight_tensor.shape)
        return self.weight_tensor



class SgdWithMomentum:  # size 改形状

    def __init__(self, global_delta, mu):
        self.global_delta = global_delta
        self.mu = mu
        self.v = None
        # self.v = np
        # self.k = 0
        self.weight_tensor = None

    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        self.weight_tensor = copy(weight_tensor)
        print(type(self.weight_tensor))
        delta = self.global_delta * individual_delta
        # self.k += 1

        if(self.v is None):
            self.v = np.zeros_like(gradient_tensor)


        if(type(self.v) == np.ndarray):
            if (type(self.v) == float):
                self.v=0
            else:
                # if(self.v.size!= gradient_tensor.size):
                self.v = np.zeros_like(gradient_tensor)


        self.v = self.mu * self.v - delta * gradient_tensor
        self.weight_tensor = self.weight_tensor + self.v
        # print('self.weight_tensor ',self.weight_tensor)
        return self.weight_tensor


class Adam:

    def __init__(self, global_delta, mu, rho):
        self.global_delta = global_delta
        self.mu = mu
        self.rho = rho
        # self.epsilon = epsilon
        self.epsilon=1e-8
        self.v = None
        self.r = None
        self.k = 0
        self.weight_tensor = None

    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        self.weight_tensor = copy(weight_tensor)
        delta = self.global_delta * individual_delta

        # print('gradient_tensor',type(gradient_tensor))
        if(self.v is None):
            if isinstance(gradient_tensor, float):
                self.v = 0
            else:
                self.v = np.zeros_like(gradient_tensor)

        if(self.r is None):
            if isinstance(gradient_tensor, float):
                self.r = 0
            else:
                self.r = np.zeros_like(gradient_tensor)


        if isinstance(gradient_tensor, np.ndarray):
            if (self.v.size != gradient_tensor.size):
                self.v = np.zeros_like(gradient_tensor)
            if (self.r.size != gradient_tensor.size):
                self.r = np.zeros_like(gradient_tensor)

        self.k += 1
        # print('self.v',self.v.shape,type(self.v))
        # print('self.r',self.r.shape,type(self.r))
        # print('gradient_tensor',gradient_tensor.shape,type(gradient_tensor))
        # print('weight_tensor',weight_tensor.shape)

        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor * gradient_tensor

        v_hat = self.v / (1 - math.pow(self.mu, self.k))
        r_hat = self.r / (1 - math.pow(self.rho, self.k))
        self.weight_tensor = self.weight_tensor - delta * ((v_hat + self.epsilon) / (np.sqrt(r_hat) + self.epsilon))
        return self.weight_tensor
