# coding: utf-8
# 2020/인공지능/final/B411001/강도연
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


def standardize(dataset):
    dataset = dataset.astype(np.float32)
    dataset -= np.mean(dataset, axis=0)
    dataset /= np.std(dataset, axis=0)
    return dataset

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

class Adamax:
    def __init__(self, lr):
        self.lr = lr
        self.m = None
        self.v = None
        self.b1 = 0.9
        self.b2 = 0.999
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.t += 1
            self.m[key] = (self.b1 * self.m[key]) + ((1 - self.b1) * grads[key])
            self.v[key] = np.maximum((self.b2 * self.v[key]), np.abs(grads[key]))
            params[key] -= (self.lr / (1 - self.b1 ** self.t)) * (self.m[key] / (self.v[key] + 1e-8))

class Model:
    """
    네트워크 모델 입니다.

    """
    def __init__(self, lr=0.002):
        """
        클래스 초기화
        """
        self.lr = lr
        self.layer = {}
        self.params = {}
        self.__init_weight()
        self.__init_layer()
        self.optimizer = Adamax(lr)

    def __init_layer(self):
        layer = {}


        layer['L1_Affine'] = Affine(self.params['L1_W'], self.params['L1_b'])
        layer['L1_ReLU'] = ReLU()


        layer['L2_Affine'] = Affine(self.params['L2_W'], self.params['L2_b'])
        layer['L2_ReLU'] = ReLU()


        layer['L3_Affine'] = Affine(self.params['L3_W'], self.params['L3_b'])
        layer['L3_ReLU'] = ReLU()

        layer['L4_Affine'] = Affine(self.params['L4_W'], self.params['L4_b'])
        layer['L4_ReLU'] = ReLU()

        layer['L5_Affine'] = Affine(self.params['L5_W'], self.params['L5_b'])
        layer['L5_softmaxWithLoss'] = SoftmaxWithLoss()

        self.layer = layer

    def __init_weight(self,):             # He 초깃값

        self.params['L1_W'] = np.random.randn(6, 13) * np.sqrt(2 / 6)
        self.params['L1_b'] = np.random.randn(1, 13)

        self.params['L2_W'] = np.random.randn(13, 27) * np.sqrt(2 / 13)
        self.params['L2_b'] = np.random.randn(1, 27)

        self.params['L3_W'] = np.random.randn(27, 55) * np.sqrt(2 / 27)
        self.params['L3_b'] = np.random.randn(1, 55)

        self.params['L4_W'] = np.random.randn(55, 111) * np.sqrt(2 / 55)
        self.params['L4_b'] = np.random.randn(1, 111)

        self.params['L5_W'] = np.random.randn(111, 6) * np.sqrt(2 / 111)
        self.params['L5_b'] = np.random.randn(1, 6)

    def update(self, x, t):
        grads = self.gradient(x, t)
        self.optimizer.update(self.params, grads)

    def predict(self, x):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """
        x = standardize(x)
        print(x[0], '나야나~ 프리딕트')
        for layer, func in self.layer.items():                                                  # layer 마다 forward를 해주는 친구
            if layer == 'L5_softmaxWithLoss':
                x = softmax(x)
            else:
                x = func.forward(x)
        return x

    def loss(self, x, t):
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        y = self.predict(x)  # 마지막에서 forward를 통해 loss를 구해준다.
        return self.layer['L5_softmaxWithLoss'].forward(y, t)

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
        x = standardize(x)
        print(x[0], '나야나~ 그래드')

        forward_L1 = self.layer['L1_ReLU'].forward(self.layer['L1_Affine'].forward(x))

        forward_L2 = self.layer['L2_ReLU'].forward(self.layer['L2_Affine'].forward(forward_L1))

        forward_L3 = self.layer['L3_ReLU'].forward(self.layer['L3_Affine'].forward(forward_L2))

        forward_L4 = self.layer['L4_ReLU'].forward(self.layer['L4_Affine'].forward(forward_L3))

        self.layer['L5_softmaxWithLoss'].forward(self.layer['L5_Affine'].forward(forward_L4), t)

        # backward & 결과 저장
        grads = {}

        backprop_L5 = self.layer['L5_Affine'].backward(self.layer['L5_softmaxWithLoss'].backward())
        grads['L5_W'] = self.layer['L5_Affine'].dW
        grads['L5_b'] = self.layer['L5_Affine'].db

        backprop_L4 = self.layer['L4_Affine'].backward(self.layer['L4_ReLU'].backward(backprop_L5))
        grads['L4_W'] = self.layer['L4_Affine'].dW
        grads['L4_b'] = self.layer['L4_Affine'].db

        backprop_L3 = self.layer['L3_Affine'].backward(self.layer['L3_ReLU'].backward(backprop_L4))
        grads['L3_W'] = self.layer['L3_Affine'].dW
        grads['L3_b'] = self.layer['L3_Affine'].db

        backprop_L2 = self.layer['L2_Affine'].backward(self.layer['L2_ReLU'].backward(backprop_L3))
        grads['L2_W'] = self.layer['L2_Affine'].dW
        grads['L2_b'] = self.layer['L2_Affine'].db

        backprop_L1 = self.layer['L1_Affine'].backward(self.layer['L1_ReLU'].backward(backprop_L2))
        grads['L1_W'] = self.layer['L1_Affine'].dW
        grads['L1_b'] = self.layer['L1_Affine'].db

        i = 1
        for key in self.params.keys():
            if key != 'L' + str(i) + '_b':
                grads[key] = (grads[key] / x.shape[0]) + (0.000009 * self.params[key])
            else:
                i += 1

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
        self.__init_layer()
