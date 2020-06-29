# coding: utf-8
# 2020/인공지능/final/B411001/강도연
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


def softmax(x):                                                         # softmax
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):                                           # cross_entropy_error
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class ReLU:                                                                # ReLU with forward & backward
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


class Affine:                                                               # Affine with forward & backward
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


class SoftmaxWithLoss:                                      # SoftmaxWithLoss forward & backward
    def __init__(self):
        self.loss = None
        self.y, self.t = None, None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


class Adamax:         # Diederik P. Kingma, ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION 논문을 참고하여 직접 작성하였습니다.
    def __init__(self, lr):         # Adamax: learning rate를 L2 norm 기반으로 조절하는 것이 아닌, infinity norm으로 조절.
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
            self.t += 1                #epoch count하기 위한 변수
            self.m[key] = (self.b1 * self.m[key]) + ((1 - self.b1) * grads[key])  # moment vector
            self.v[key] = np.maximum((self.b2 * self.v[key]), np.abs(grads[key]))   # infinity norm 이용
            params[key] -= (self.lr / (1 - self.b1 ** self.t)) * (self.m[key] / (self.v[key] + 1e-8))


class Model:
    """
    네트워크 모델 입니다.

    """
    def __init__(self, lr=0.002):           # 설계한 모델에서, test를 통해 찾은 최적의 lr인 0.002를 default로 설정하였습니다.
        self.lr = lr
        self.layer = {}
        self.params = {}
        self.__init_weight()
        self.__init_layer()
        self.optimizer = Adamax(lr)

    def __init_layer(self):          # layer 뼈대 구성
        layer = {}

        layer['L1_Affine'] = Affine(self.params['L1_W'], self.params['L1_b'])
        layer['L1_ReLU'] = ReLU()

        layer['L2_Affine'] = Affine(self.params['L2_W'], self.params['L2_b'])
        layer['L2_ReLU'] = ReLU()

        layer['L3_Affine'] = Affine(self.params['L3_W'], self.params['L3_b'])
        layer['L3_ReLU'] = ReLU()

        layer['L4_Affine'] = Affine(self.params['L4_W'], self.params['L4_b'])
        layer['L4_ReLU'] = ReLU()                                              # 훈련을 위해 ReLu activation 사용

        layer['L5_Affine'] = Affine(self.params['L5_W'], self.params['L5_b'])
        layer['L5_softmaxWithLoss'] = SoftmaxWithLoss()                        # output을 위해 softmax activation 사용

        self.layer = layer

    def __init_weight(self,):       # 각 layer 사이사이의 W, b 값들의 초기화. 이때, ReLU를 썻기때문에 이에 맞는 He 초깃값 사용
        self.params['L1_W'] = np.random.randn(6, 13) * np.sqrt(2 / 6)                               # layer2 unit은 13개
        b1 = np.random.randn(1, 1)
        self.params['L1_b'] = np.full((1, 13), b1)

        self.params['L2_W'] = np.random.randn(13, 27) * np.sqrt(2 / 13)                             # layer3 unit은 27개
        b2 = np.random.randn(1, 1)
        self.params['L2_b'] = np.full((1, 27), b2)

        self.params['L3_W'] = np.random.randn(27, 55) * np.sqrt(2 / 27)                             # layer4 unit은 55개
        b3 = np.random.randn(1, 1)
        self.params['L3_b'] = np.full((1, 55), b3)

        self.params['L4_W'] = np.random.randn(55, 111) * np.sqrt(2 / 55)                           # layer5 unit은 111개
        b4 = np.random.randn(1, 1)
        self.params['L4_b'] = np.full((1, 111), b4)

        self.params['L5_W'] = np.random.randn(111, 6) * np.sqrt(2 / 111)
        b5 = np.random.randn(1, 1)
        self.params['L5_b'] = np.full((1, 6), b5)

    def update(self, x, t):                      # backpropagation을 통한 gradient계산 후, optimizer를 통해 params update
        grads = self.gradient(x, t)
        self.optimizer.update(self.params, grads)

    def predict(self, x):
        for layer, func in self.layer.items():      # 설계된 network에서 예측 값 산출위해 forward들을 진행 후, 마지막에 softmax 적용
            if layer == 'L5_softmaxWithLoss':
                x = softmax(x)
            else:
                x = func.forward(x)
        return x

    def loss(self, x, t):      # predict()를 통해 나온 예측 값과, target값을 이용하여 cross_entropy_error를 통한 error값 계산
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def gradient(self, x, t):

        # forward ( Input에 대해 Affine, ReLU 적용. 마지막 layer에는 softmax를 통한 예측 값과, loss값 계산)
        forward_L1 = self.layer['L1_ReLU'].forward(self.layer['L1_Affine'].forward(x))

        forward_L2 = self.layer['L2_ReLU'].forward(self.layer['L2_Affine'].forward(forward_L1))

        forward_L3 = self.layer['L3_ReLU'].forward(self.layer['L3_Affine'].forward(forward_L2))

        forward_L4 = self.layer['L4_ReLU'].forward(self.layer['L4_Affine'].forward(forward_L3))

        self.layer['L5_softmaxWithLoss'].forward(self.layer['L5_Affine'].forward(forward_L4), t)

        # backward ( 마지막 layer에서부터 backward를 통한 gradient값 계산 및 저장)
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

        # Overfitting 방지를 위해 weight decay 적용, weight decay parameter는 0.000009
        i = 1
        for key in self.params.keys():
            if key != 'L' + str(i) + '_b':
                grads[key] = (grads[key] / x.shape[0]) + (0.000009 * self.params[key])
            else:
                i += 1

        # 최종 gradient값들 return
        return grads

    def save_params(self, file_name="params.pkl"):        # training 후 params 정보를 담은 params.pkl 파일 저장
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):      # test시 가져올 params.pkl 파일 load.
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        self.__init_layer()                              # load 후, 가져온 params 정보로 layer 다시 구성.
