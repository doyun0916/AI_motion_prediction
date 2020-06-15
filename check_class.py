# coding: utf-8
# 2020/인공지능/final/B411001/강도연
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx


class Affine:
    def __init__(self, W, b):
        self.W, self.b = W, b
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # error값을 저장하면 쓸데가 있다.
        self.y, self.t = None, None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
        # softmax와 cross_entropy_error는 예전에 쓰던거 썻다 가정

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


class Adagrad:
    def __init__(self, lr):
        self.lr = lr
        self.h = None
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Model:
    """
    네트워크 모델 입니다.

    """
    def __init__(self, lr=0.01):
        """
        클래스 초기화
        """

        self.params = {}
        self.__init_weight()
     #   self.__init_layer()
        self.optimizer = Adagrad(lr)

    #def __init_layer(self):


    def __init_weight(self,):             # He 초깃값

        self.params['L1_W'] = np.random.randn(6, 10) * np.sqrt(2 / 6)
        self.params['L1_b'] = np.random.randn(1, 10)
        for i in range(3):
            self.params['L' + str(i + 2) + '_W'] = np.random.randn(10, 10) * np.sqrt(2 / 10)
            self.params['L' + str(i + 2) + '_b'] = np.random.randn(1, 10)
        self.params['L5_W'] = np.random.randn(10, 6) * np.sqrt(2 / 10)
        self.params['L5_b'] = np.random.randn(1, 6)

    def update(self, x, t):
        print(self.params['L1_W'])
        grads = self.gradient(x, t)
        self.optimizer.update(self.params, grads)

    def predict(self, x):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """
        for layer in self.layers.values():                                                  # layer 마다 forward를 해주는 친구
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        y = self.predict(x)                                                      # 마지막에서 forward를 통해 loss를 구해준다.
        return self.last_layer.forward(y, t)


    def gradient(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구하는 함수입니다.
        첫번째로 받은데이터를 forward propagation 시키고,
        두번째로 back propagation 시켜 grads에 미분값을 리턴합니다.
        :param x: data
        :param t: data_label
        :return: grads
        """
        # forward
        affine_L1 = Affine(self.params['L1_W'], self.params['L1_b'])
        relu_L1 = ReLU()

        forward_L1 = relu_L1.forward(affine_L1.forward(x))

        affine_L2 = Affine(self.params['L2_W'], self.params['L2_b'])
        relu_L2 = ReLU()

        forward_L2 = relu_L2.forward(affine_L2.forward(forward_L1))

        affine_L3 = Affine(self.params['L3_W'], self.params['L3_b'])
        relu_L3 = ReLU()

        forward_L3 = relu_L3.forward(affine_L3.forward(forward_L2))

        affine_L4 = Affine(self.params['L4_W'], self.params['L4_b'])
        relu_L4 = ReLU()

        forward_L4 = relu_L4.forward(affine_L4.forward(forward_L3))

        affine_L5 = Affine(self.params['L5_W'], self.params['L5_b'])
        softwithloss = SoftmaxWithLoss()

        softwithloss.forward(affine_L5.forward(forward_L4), t)


        # backward & 결과 저장
        grads = {}
        backprop_L6 = affine_L5.backward(softwithloss.backward())
        grads['L5_W'] = affine_L5.dW
        grads['L5_b'] = affine_L5.db

        backprop_L5 = affine_L4.backward(relu_L4.backward(backprop_L6))
        grads['L4_W'] = affine_L4.dW
        grads['L4_b'] = affine_L4.db

        backprop_L4 = affine_L3.backward(relu_L3.backward(backprop_L5))
        grads['L3_W'] = affine_L3.dW
        grads['L3_b'] = affine_L3.db

        backprop_L3 = affine_L2.backward(relu_L2.backward(backprop_L4))
        grads['L2_W'] = affine_L2.dW
        grads['L2_b'] = affine_L2.db

        backprop_L2 = affine_L1.backward(relu_L1.backward(backprop_L3))
        grads['L1_W'] = affine_L1.dW
        grads['L1_b'] = affine_L1.db

        return grads

    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        pass
