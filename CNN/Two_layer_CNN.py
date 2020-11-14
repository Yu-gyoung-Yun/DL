#!/usr/bin/env python
# coding: utf-8

# In[8]:
import numpy as np
from functions.NN_funct import *

class Two_layer_CNN: # Conv-ReLU-MaxPool - Conv-ReLU-MaxPool -Linear-SoftMax (The input and output size of NN: input 28x28, output 10)
    def __init__(self, filter_num = 20, filter=(5,5), pad = 0, stride = 1):
        self.input_dim = 28
        self.filter_num = filter_num
        self.filter_size = filter
        self.pad = pad
        self.stride = stride
        print(self.filter_size)
        self.output_height = (self.input_dim + 2*pad - self.filter_size[0]) / stride +1
        self.output_width = (self.input_dim + 2*pad - self.filter_size[1]) / stride + 1
        #(0,0,28,28)에서 0,0은 임의의 값.함수사용하기 위해 아무숫자넣은 것.

        self.pool_output = int(filter_num*2 * (self.output_height/2) ** 2) #w2가 filter_num*2이므로 그 출력값에 맞춰서

        self.weights = {}
        self.weights['w1'] = np.random.randn(filter_num, 1, self.filter_size[0], self.filter_size[1]) #일단 filter 직사각형도 허용...!!!!
        self.weights['b1'] = np.random.randn(filter_num)
        self.weights['w2'] = np.random.randn(filter_num*2, 1, self.filter_size[0]*2, self.filter_size[1]*2) #일단 그냥 *2해봤음. .. 맞는지 모르겠음!!!!
        self.weights['b2'] = np.random.randn(filter_num*2)
        self.weights['w3'] = np.random.randn(self.pool_output, 50) #hidden_size = 50
        self.weights['b3'] = np.random.randn(50)

        self.layers = {}
        self.layers['CL1'] = Conv(self.weights['w1'], self.weights['b1'], self.stride, self.pad)
        self.layers['RL1'] = ReLU()
        self.layers['MP1'] = Maxpool((2,2),stride = 1,pad = 0)
        self.layers['CL2'] = Conv(self.weights['w2'], self.weights['b2'], self.stride, self.pad)
        self.layers['RL2'] = ReLU()
        self.layers['MP2'] = Maxpool((2,2),stride = 1,pad = 0)
        self.layers['L1'] = Linear(self.weights['w3'], self.weights['b3'])
        self.layers['softmax_cross'] = Softmax_Cross_Entropy_Error()

        self.l = ['CL1','RL1','MP1','CL2','RL2','MP2','L1','softmax_cross']

    def forward(self, x):
        for layer in self.l:
            if layer != 'softmax_cross':
                x = self.layers[layer].forward(x)
        return x

    def loss(self, x, y):
        f = self.forward(x)
        loss = self.layers['softmax_cross'].forward(f)
        return loss

    def accuracy(self, x, y):
        f = self.forward(x)
        p = np.argmax(f, axis = 1) #argmax는 가장 큰 값의 인덱스 값을 반환한다.
        #one_hot encoding이니까 if문 적용 x.!!
        y = np.argmax(y, axis=1) #행 (1,batch_size)
        accuracy = np.sum(y == p) / float(x.shape[0]) #batch_size로 나눠
        return accuracy

    def back_propagate_train(self,x,y):
        loss = self.loss(x,y)
        back = self.layers['softmax_cross'].backward(1)
        for reversed_layer in reversed(self.l):
            back = self.layers[reversed_layer].backward(back)

        gradients = {}
        gradients['w1'] = self.layers['CL1'].d_filter
        gradients['b1'] = self.layers['CL1'].db
        gradients['w2'] = self.layers['CL2'].d_filter
        gradients['b2'] = self.layers['CL2'].db
        gradients['w3'] = self.layers['L1'].dw
        gradients['b3'] = self.layers['L1'].db

        return gradients


# In[ ]:




