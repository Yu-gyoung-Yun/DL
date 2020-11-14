#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np

class ReLU:
    def __init__(self):
        self.x_bool = None

    def forward(self,x):
         #print('ReLU_forward')
        self.x_bool = (x<=0) #T/F bool Array
        relu = x.copy()
        relu[self.x_bool] = 0
        return relu

    def backward(self,back):
        #print('ReLU_backward')
        back[self.x_bool] = 0
        return back 

class LReLU:
    def __init__(self):
        self.x_bool = None
  
    def forward(self,x):
        self.x_bool = (x<=0)
        lrelu = x.copy()
        lrelu[self.x_bool] *= 0.1
        return lrelu

    def backward(self,back):
        back[self.x_bool] *= 0.1
        return back

class Softmax_Cross_Entropy_Error:
    def __init__(self):
        self.loss = 0
        self.p = 0
        self.y = 0

    def forward(self, p, y):
        #print('Softmax_Cross_Entropy_Error_forward')
        self.y = y
        self.p = softmax(p)
        self.loss = cross_entropy_error(self.p, self.y)
        return self.loss

    def backward(self, back = 1):
        #print('Softmax_Cross_Entropy_Error_backward')
        batch_size = self.y.shape[0]
        return (self.p - self.y) / batch_size
    
class Linear:
    def __init__(self,w,b):
        self.w = w
        self.b = b
        self.dw = 0
        self.db = 0
        self.x = 0

    def forward(self,x):
        #print('Linear_forward')
        self.x = x
        return np.dot(x, self.w)+self.b

    def backward(self,back):
        #print('Linear_backward')
        self.dw = np.dot(self.x.T, back)
        self.db = np.sum(back, axis=0)

        return np.dot(back, self.w.T)


# In[2]:


#For Convolution Network
def cal_output(input_img, filter, stride, pad):
    output_height = (input_img.shape[2] + 2*pad - filter.shape[2]/stride) + 1
    output_width = (input_img.shape[3] + 2*pad - filter.shape[3]/stride) + 1

    return output_height, output_width


# In[4]:


def im2col(input_image, filter, stride=1, pad=0):
    output_height, output_width = cal_output(input_image, filter, stride, pad)

    output_col = np.zeros((int(input_image.shape[0]), int(input_image.shape[1]), int(filter.shape[2]), int(filter.shape[3]), int(output_height), int(output_width)))
    pad_img = np.pad(input_image, [(0,0),(0,0),(pad,pad),(pad,pad)], 'constant')

    for i in range(filter.shape[2]):#각 filter의 원소마다 연산해 줄 이미지 추출
        y_max = i + stride * output_height
        for j in range(filter.shape[3]):
            x_max = j + stride * output_width
            output_col[:, :, i, j, :, :] = pad_img[:, :, i: y_max: stride, j:x_max:stride]
    output_col = output_col.transpose(0,4,5,1,2,3).reshape(input_image.shape[0]*output_height*output_width,-1)
    return output_col

#https://cding.tistory.com/112
def col2im(col, input_image, filter, stride=1, pad=0):
    output_height, output_width = cal_output(input_image, filter, stride, pad)
    col = col.reshape(int(input_image.shape[0]), int(output_height), int(output_width), int(input_image.shape[1]), int(filter.shape[2]), int(filter.shape[3])).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((int(input_image.shape[0]), int(input_image.shape[1]), int(input_image.shape[2]) + 2 * pad + stride - 1, int(input_image.shape[3]) + 2 * pad + stride - 1))
    for y in range(filter.shape[2]):
        y_max = y + stride * output_height
        for x in range(filter.shape[3]):
            x_max = x + stride * output_width
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:input_image.shape[2] + pad, pad:input_image.shape[3] + pad]


class Maxpool:
    def __init__(self, pool_size ,stride,pad): #stride, pad 초기화 안함
        self.pool = pool_size #(2,2) tuple형태, tuple도 indexing 가능
        self.stride = stride
        self.pad = pad
        self.col_max = None
        self.x = None

    def forward(self, x):
        self.x = x
        output_height, output_width = cal_output(x, self.pool, self.stride, pad = 0)
        _col = im2col(x, self.pool, self.stride, self.pad).reshape(-1, self.pool[0]*self.pool[1])

        col_max = np.max(_col, axis = 1) #행단위로 max찾아
        self.col_max = np.argmax(_col, axis = 1) #max의 index값
        return col_max.reshape(x.shape[0], output_height, output_width, x.shape[1]).transpose(0,3,1,2)
  
    def backward(self, back): #이름만 바꿨으므로 수정필요

        back = back.transpose(0, 2, 3, 1)

        pool_size = self.pool[0] * self.pool[1]
        dmax = np.zeros((back.size, pool_size))
        dmax[np.arange(self.col_max.size), self.col_max.flatten()] = back.flatten()
        dmax = dmax.reshape(back.shape + (pool_size,)) 

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool, self.stride, self.pad)
        return dx


class Conv: #filter가 정사각, 직사각 둘다 가능
    def __init__(self, filter_weights, bias, stride = 1, pad = 0):
        print('__init__COnv')
        self.filter = filter_weights
        self.bias = bias
        self.stride = stride
        self.pad = pad
        self.d_filter = None
        self.db = None
        self.input = None
        self.output = None

    def forward(self, x):
        self.x = x
        output_height = (x.shape[2] + 2*self.pad - self.filter.shape[2]/self.stride) + 1
        output_width = (x.shape[3] + 2*self.pad - self.filter.shape[3]/self.stride) + 1

        image_col = im2col(x, self.filter, self.stride, self.pad)
        self.input = image_col
        filter_col = self.filter.reshape(self.filter.shape[0],-1).T
        self.output= filter_col
        out_col = np.dot(image_col,filter_col) + self.bias

        return out_col.reshape(x.shape[0],output_height, output_width, -1).transpose(0,3,1,2)

    def backward(self, back):
        self.d_filter = np.dot(self.input.T, back).transpose(1, 0).reshape(self.filter.shape[0],self.filter.shape[1] , self.filter.shape[2], self.filter.shape[3])
        self.db = np.sum(back, axis=0)
        back_propagation = np.dot(back, self.output.T)

        return col2im(back_propagation, self.x, self.filter, self.stride, self.pad) #col2im 함수 내가 만들어야해(input달라)


# In[ ]:




