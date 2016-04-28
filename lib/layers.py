import os.path
import sys
import logging
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn
from theano.sandbox.cuda.basic_ops import gpu_contiguous
# from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
# from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from pylearn2.expr.normalize import CrossChannelNormalization

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import warnings
warnings.filterwarnings("ignore")

from config import random_seed

rng = np.random.RandomState(random_seed)
srng = RandomStreams(random_seed)


class Layer:
    params = None

    def __init__(self, input_layer, fun=None, params=None, name=None, fixed=False):
        self.input_layer = input_layer
        self.n_channel = self.input_layer.n_channel
        self.fun = fun
        self.params = params
        self.name = name
        self.fixed = fixed
        # self.input_layer=None
#        self._shape = input_layer.shape()

    def output(self,   *args, **kwargs):
        assert not None == self._output
        assert not self == self.input_layer
        input = self.input_layer.output(*args, **kwargs)
        return self._output(input=input,  *args, **kwargs)

    def _output(self, input,   *args, **kwargs):
        if not None == self.params:
            return self.fun(input, self.params)
        return self.fun(input)
#    def shape(self):
#        return self._shape

    def weight(self):
        l = []
        if not None == self.params:
            if not self.fixed:
                l = self.params.values()
        return l + self.input_layer.weight()

    def bias(self):
        return self.input_layer.bias()

    def getParams4SL(self, count):
        l = []
        if not None == self.params:
            assert not None == self.name
            l = [(self.name + '_' + k, v) for k, v in self.params.items()]
        return l, count

    def saveParams(self, path, count=0):
        if not os.path.exists(path):
            os.makedirs(path)
        params, count = self.getParams4SL(count)
        diffs = self.input_layer.saveParams(path, count)
        for n, p in params:
            n = os.path.join(path, n) + '.npy'
            data = p.get_value()
            if os.path.exists(n):
                data_old = np.load(n)
                diff = abs(data - data_old)
                diffs += [(n, np.mean(diff), np.mean(diff**2), np.max(diff))]
            np.save(n, data)
        return diffs

    def loadParams(self, path, count=0):
        params, count = self.getParams4SL(count)
        self.input_layer.loadParams(path, count)
        for n, p in params:
            n = os.path.join(path, n) + '.npy'
            if os.path.exists(n):
                p.set_value(np.load(n))
            else:
                print('parameters file %s not found' % n)


class DataLayer:

    def __init__(self,  n_channel=3):  # , shape):
       # self.data = data
        self.n_channel = n_channel
#        self._shape = shape

    def output(self, data_layer,  *args, **kwargs):
        return data_layer
        return self.data

#    def shape(self):
#        return self._shape

    def weight(self):
        return []

    def bias(self):
        return []

    def saveParams(self, path, count=0):
        return []

    def loadParams(self, path, count=0):
        return


class Inception(Layer):

    def __init__(self, input_layer,  n_in,
                 n_out_1x1=0,
                 n_out_3x3=0,
                 n_out_3x3_r=0,
                 n_out_5x5=0,
                 n_out_5x5_r=0,
                 n_out_7x7=0,
                 n_out_7x7_r=0,
                 n_out_mp_r=0, bias=0, std=0.01, fixed=False, name=None):
        if n_out_3x3 < 1:
            n_out_3x3_r = 0
        if n_out_5x5 < 1:
            n_out_5x5_r = 0
        if n_out_7x7 < 1:
            n_out_7x7_r = 0

        self.input_layer = input_layer
        self.fixed = fixed
        self.name = name

        def getParams(n_out, n_in=n_in, size=1):
            if n_out < 1 or n_in < 1:
                return None, None
            w = generateWeight((n_out, n_in, size, size,), std=std)
            b = generateWeight(n_out, init=bias, std=0)
            return w, b

        self.W_1x1, self.b_1x1 = getParams(n_out_1x1)
        self.W_3x3_r, self.b_3x3_r = getParams(n_out_3x3_r)
        self.W_5x5_r, self.b_5x5_r = getParams(n_out_5x5_r)
        self.W_7x7_r, self.b_7x7_r = getParams(n_out_7x7_r)
        self.W_mp_r, self.b_mp_r = getParams(n_out_mp_r)

        n_in_ = n_in if n_out_3x3_r < 1 else n_out_3x3_r
        self.W_3x3, self.b_3x3 = getParams(n_out_3x3, n_in_, 3)
        n_in_ = n_in if n_out_5x5_r < 1 else n_out_5x5_r
        self.W_5x5, self.b_5x5 = getParams(n_out_5x5, n_in_, 5)
        n_in_ = n_in if n_out_7x7_r < 1 else n_out_7x7_r
        self.W_7x7, self.b_7x7 = getParams(n_out_7x7, n_in_, 7)

    def _output(self, input,  *args, **kwargs):
        results = []

        def conv(w, b, pad=0, img=None):
            if None == img:
                img = input
            if None == w:
                return None
            out = dnn.dnn_conv(
                img=img, kerns=w, subsample=(1, 1), border_mode=pad)
            out += b.dimshuffle('x', 0, 'x', 'x')
            out = T.maximum(out, 0)
            return out

        r_1 = None
        if not None == self.W_1x1:
            r_1 = conv(self.W_1x1, self.b_1x1)
            if not self.W_mp_r == None:
                r_1 = r_1[:, :, 1:-1, 1:-1]
        pad_ = 1 if self.W_mp_r == None else 0

        r_3_r = conv(self.W_3x3_r, self.b_3x3_r)
        r_3 = conv(self.W_3x3, self.b_3x3, img=r_3_r, pad=pad_ + 0)
        r_5_r = conv(self.W_5x5_r, self.b_5x5_r)
        r_5 = conv(self.W_5x5, self.b_5x5, img=r_5_r, pad=pad_ + 1)
        r_7_r = conv(self.W_7x7_r, self.b_7x7_r)
        r_7 = conv(self.W_7x7, self.b_7x7, img=r_7_r, pad=pad_ + 2)
        r_mp = None
        if not None == self.W_mp_r:
            r_mp = dnn.dnn_pool(input, ws=(3, 3), stride=(1, 1))
            r_mp = conv(self.W_mp_r, self.b_mp_r, img=r_mp)
        for r in [r_1, r_3, r_5, r_7, r_mp]:
            if not None == r:
                results.append(r)
        r = T.concatenate(results, axis=1)
        return r

    def getParams4SL(self, count):
        r = []
        if None == self.name:
            prefix = 'incep%.3d_' % count
            count += 1
        else:
            prefix = self.name + '_'

        for n in self.__weights + self.__biases:
            p = getattr(self, n)
            if p == None:
                continue
            r.append((prefix + n, p))
        return r, count

    __weights = ['W_1x1', 'W_3x3', 'W_5x5', 'W_7x7',
                 'W_3x3_r', 'W_5x5_r', 'W_7x7_r', 'W_mp_r']

    def weight(self):
        if self.fixed:
            return [] + self.input_layer.weight()
        r = []
        for w in self.__weights:
            w = getattr(self, w)
            if not None == w:
                r.append(w)
        return r + self.input_layer.weight()

    __biases = ['b_1x1', 'b_3x3', 'b_5x5', 'b_7x7',
                'b_3x3_r', 'b_5x5_r', 'b_7x7_r', 'b_mp_r']

    def bias(self):
        if self.fixed:
            return [] + self.input_layer.bias()
        r = []
        for b in self.__biases:
            b = getattr(self, b)
            if not None == b:
                r.append(b)
        return r + self.input_layer.bias()


class Inception1(Layer):

    def __init__(self, input_layer,
                 n_out_3x3_r=0,
                 n_out_5x5_r=0,
                 n_out_7x7_r=0,
                 bias=0, std=0.01, fixed=False, name=None):
        n_out_1x1 = n_out_3x3_r + n_out_5x5_r + n_out_7x7_r
        n_in = input_layer.n_channel
        self.affected_channels = n_out_1x1
        self.n_out_3x3_r = n_out_3x3_r
        self.n_out_5x5_r = n_out_5x5_r
        self.n_out_7x7_r = n_out_7x7_r
        self.n_in = n_in

        self.input_layer = input_layer
        self.fixed = fixed
        self.name = name

        if n_out_1x1 < 1:
            self.fixed = True
        self.n_channel = n_out_1x1 + n_in

        self.W = generateWeight((n_out_1x1, n_in, 1, 1,), std=std)
        self.b = generateWeight(n_out_1x1, init=bias, std=0)

    def _output(self, input,  *args, **kwargs):
        if self.n_channel == self.n_in:
            return input
        out = dnn.dnn_conv(
            img=input, kerns=self.W, subsample=(1, 1), border_mode=0)
        out += self.b.dimshuffle('x', 0, 'x', 'x')
        r = T.concatenate([out, input], axis=1)
        return r

    def getParams4SL(self, count):
        if None == self.name:
            prefix = 'incep%.3d_' % count
            count += 1
        else:
            prefix = self.name + '_'
        r = [(prefix + 'W', self.W), (prefix + 'b', self.b)]
        return r, count

    def weight(self):
        r = [] if self.fixed else [self.W]
        return r + self.input_layer.weight()

    def bias(self):
        r = [] if self.fixed else [self.b]
        return r + self.input_layer.bias()


class Inception2(Layer):

    def __init__(self, input_layer,
                 n_out_1x1=0,
                 n_out_3x3=0,
                 n_out_5x5=0,
                 n_out_7x7=0,
                 n_out_mp_r=0, bias=0, std=0.01, fixed=False, name=None):
        incep1 = input_layer
        while not incep1.__class__ == Inception1:
            incep1 = incep1.input_layer
        n_in = incep1.n_in
        n_out_3x3_r = incep1.n_out_3x3_r
        n_out_5x5_r = incep1.n_out_5x5_r
        n_out_7x7_r = incep1.n_out_7x7_r

        self._bias = bias
        self._std = std

        self._n_in = n_in
        self.n_in = n_in

        self.n_out_3x3_r = n_out_3x3_r
        self.n_out_5x5_r = n_out_5x5_r
        self.n_out_7x7_r = n_out_7x7_r

        self.input_layer = input_layer
        self.fixed = fixed
        self.name = name

        self.n_channel = n_out_1x1 + n_out_3x3 + \
            n_out_5x5 + n_out_7x7 + n_out_mp_r

        self.W_1x1, self.b_1x1 = self._genParams(n_out_1x1)
        self.W_mp_r, self.b_mp_r = self._genParams(n_out_mp_r)

        n_in_ = n_in if n_out_3x3_r < 1 else n_out_3x3_r
        self.W_3x3, self.b_3x3 = self._genParams(n_out_3x3, n_in=n_in_, size=3)
        n_in_ = n_in if n_out_5x5_r < 1 else n_out_5x5_r
        self.W_5x5, self.b_5x5 = self._genParams(n_out_5x5, n_in=n_in_, size=5)
        n_in_ = n_in if n_out_7x7_r < 1 else n_out_7x7_r
        self.W_7x7, self.b_7x7 = self._genParams(n_out_7x7, n_in=n_in_, size=7)

    def getParams4SL(self, count):
        r = []
        if None == self.name:
            prefix = 'incep%.3d_' % count
            count += 1
        else:
            prefix = self.name + '_'

        for n in self.__weights + self.__biases:
            p = getattr(self, n)
            if p == None:
                continue
            r.append((prefix + n, p))
        return r, count

    def _genParams(self, n_out, std=None, bias=None, n_in=None, size=1):
        if None == std:
            std = self._std
        if None == bias:
            bias = self._bias
        if None == n_in:
            n_in = self._n_in
        if n_out < 1 or n_in < 1:
            return None, None
        w = generateWeight((n_out, n_in, size, size,), std=std)
        b = generateWeight(n_out, init=bias, std=0)
        return w, b

    def _conv(self, w, b, img, pad=0, ):
        if None == w:
            return None
        out = dnn.dnn_conv(
            img=img, kerns=w, subsample=(1, 1), border_mode=pad)
        out += b.dimshuffle('x', 0, 'x', 'x')
        return out

    def _output(self, input,  *args, **kwargs):
        origin = input[:, - self.n_in:]
        r_3_r, r_5_r, r_7_r = origin, origin, origin

        c = 0
        if self.n_out_3x3_r > 0:
            r_3_r = input[:, c:c + self.n_out_3x3_r]
            c += self.n_out_3x3_r
        if self.n_out_5x5_r > 0:
            r_5_r = input[:, c:c + self.n_out_5x5_r]
            c += self.n_out_5x5_r
        if self.n_out_7x7_r > 0:
            r_7_r = input[:, c:c + self.n_out_7x7_r]
            c += self.n_out_7x7_r
        input = origin

        r_1 = None
        if not None == self.W_1x1:
            r_1 = self._conv(self.W_1x1, self.b_1x1, img=input)
            if not self.W_mp_r == None:
                r_1 = r_1[:, :, 1:-1, 1:-1]
        pad_ = 1 if self.W_mp_r == None else 0

        r_3 = self._conv(self.W_3x3, self.b_3x3, img=r_3_r, pad=pad_ + 0)
        r_5 = self._conv(self.W_5x5, self.b_5x5, img=r_5_r, pad=pad_ + 1)
        r_7 = self._conv(self.W_7x7, self.b_7x7, img=r_7_r, pad=pad_ + 2)
        r_mp = None
        if not None == self.W_mp_r:
            r_mp = dnn.dnn_pool(input, ws=(3, 3), stride=(1, 1))
            r_mp = self._conv(self.W_mp_r, self.b_mp_r, img=r_mp)
        results = []
        for r in [r_1, r_3, r_5, r_7, r_mp]:
            if not None == r:
                results.append(r)
        r = T.concatenate(results, axis=1)
        return r
    __weights = ['W_1x1', 'W_3x3', 'W_5x5', 'W_7x7', 'W_mp_r']
    __biases = ['b_1x1', 'b_3x3', 'b_5x5', 'b_7x7', 'b_mp_r']

    def weight(self):
        r = []
        if not self.fixed:
            for w in self.__weights:
                w = getattr(self, w)
                if not None == w:
                    r.append(w)
        return r + self.input_layer.weight()

    def bias(self):
        r = []
        if not self.fixed:
            for b in self.__biases:
                b = getattr(self, b)
                if not None == b:
                    r.append(b)
        return r + self.input_layer.bias()


class ConvLayer(Layer):

    def __init__(self, input_layer,  filter_shape, stride, pad, bias, std=0.01, fixed=False, name=None, normalize_weight=None):
        #        assert input_shape == input_layer.shape()
        #        self._shape = (filter_shape[3], (input_shape[1] + 2 * pad - filter_shape[1] + stride) / stride, (
        # input_shape[2] + 2 * pad - filter_shape[2] + stride) / stride,
        # input_shape[3])
        self.n_in = filter_shape[0]
        self.n_out = filter_shape[3]
        self.n_channel = self.n_out
        self.input_layer = input_layer
        self.subsample = (stride, stride)
        self.pad = pad
        self.filter_shape = np.asarray(filter_shape)
        self.W = generateWeight(self.filter_shape, std=std)
        self.b = generateWeight(self.filter_shape[3], init=bias, std=0)
        self.fixed = fixed
        self.name = name
        self.normalize_weight = normalize_weight

    def _output(self, input,  *args, **kwargs):
        input_shuffled = input
        W_shuffled = self.W.dimshuffle(3, 0, 1, 2)
        if not None == self.normalize_weight:
            epsilon = 1e-6
            W_shuffled = W_shuffled * self.normalize_weight / \
                (T.mean(W_shuffled**2, axis=(1, 2, 3))
                 [:, np.newaxis, np.newaxis, np.newaxis] + epsilon)**.5
        conv_out = dnn.dnn_conv(img=input_shuffled,
                                kerns=W_shuffled,
                                subsample=self.subsample,
                                border_mode=self.pad)
        conv_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        out = conv_out
        return out

    def getParams4SL(self, count):

        if None == self.name:
            w = 'conv%.3d_w' % count
            b = 'conv%.3d_b' % count
            count += 1
        else:
            w = self.name + '_w'
            b = self.name + '_b'
        w += '_%dx%d' % (self.n_in, self.n_out)
        b += '_%dx%d' % (self.n_in, self.n_out)
        return [(w, self.W), (b, self.b)], count

    def weight(self):
        if self.fixed:
            return [] + self.input_layer.weight()
        return [self.W] + self.input_layer.weight()

    def bias(self):
        if self.fixed:
            return [] + self.input_layer.bias()
        return [self.b] + self.input_layer.bias()


class PReLULayer(Layer):

    # , affected_channels=None):
    def __init__(self, input_layer,  alpha=.25, name=None, fixed=False, is_conv=True):
        self.input_layer = input_layer
        self.n_channel = self.input_layer.n_channel
        affected_channels = getattr(input_layer, 'affected_channels', None)
        self.affected_channels = self.n_channel if None == affected_channels else affected_channels
        assert self.n_channel >= self.affected_channels
        if is_conv:
            self.filter_shape = 1, self.affected_channels, 1, 1
        else:
            self.filter_shape = 1, self.affected_channels
        self.alpha = generateWeight(self.affected_channels, init=alpha, std=0)
        self.name = name
        self.fixed = fixed

    def _output(self, input,  *args, **kwargs):
        k = (self.alpha - 1).reshape(self.filter_shape)
        if self.affected_channels == self.n_channel:
            return input + T.minimum(0, input) * k
        else:
            affected = input[:, :self.affected_channels]
            unaffected = input[:, self.affected_channels:]
            affected = affected + T.minimum(0, affected) * k
            return T.concatenate([affected, unaffected], axis=1)

    def getParams4SL(self, count):

        if None == self.name:
            a = 'conv%.3d_a' % count
            count += 1
        else:
            a = self.name + '_a'
        a += '_' + str(self.affected_channels)
        return [(a, self.alpha)], count

    def weight(self):
        if self.fixed:
            return [] + self.input_layer.weight()
        return [self.alpha] + self.input_layer.weight()


class GaussianNoise(Layer):

    def __init__(self, input_layer, std=1.0, mean=0):
        self.mean = mean
        self.std = std
        self.input_layer = input_layer
        self.n_channel = self.input_layer.n_channel

    def _output(self, input,  *args, **kwargs):
        r = srng.normal(size=input.shape, avg=self.mean, std=self.std,
                        dtype=theano.config.floatX)
        return input + r


class ImageNormalization(Layer):
    epsilon = 1e-6

    def _output(self, input,  *args, **kwargs):
        mean = T.mean(input, axis=(2, 3))  # , keepdims=True)
        mean2 = T.mean(input**2, axis=(2, 3))  # , keepdims=True)
        std = T.sqrt(mean2 - mean**2 + self.epsilon)
       # std = T.std(input, axis=(2, 3))   + self.epsilon
        mean = mean[:, :, np.newaxis, np.newaxis]
        std = std[:, :, np.newaxis, np.newaxis]
        return (input - mean) / std


class BatchNormalization(Layer):

    def __init__(self, input_layer, is_conv=True, noparam=False,
                 load_distribution=False,
                 epsilon=1e-6, name=None, fixed=False, beta=0, gamma=1):  # , affected_channels=None):
        self.input_layer = input_layer
        self.epsilon = epsilon
        self.noparam = noparam
        self.name = name
        self.fixed = fixed
        n_channel = self.input_layer.n_channel
        self.n_channel = n_channel
        affected_channels = getattr(input_layer, 'affected_channels', None)
        affected_channels = n_channel if None == affected_channels else affected_channels
        self.affected_channels = affected_channels
        self.is_conv = is_conv
        self.load_distribution = load_distribution
        self.dst_mean = generateWeight(affected_channels, init=0, std=0)
        self.dst_std = generateWeight(affected_channels, init=1, std=0)
        if noparam:
            self.fixed = True
            self.beta = beta
            self.gamma = gamma
        else:
            self.beta = generateWeight(affected_channels, init=beta, std=0)
            self.gamma = generateWeight(affected_channels, init=gamma, std=0)

    def getDistribution(self, input):
        if self.load_distribution:
            mean = self.dst_mean
            std = self.dst_std
        else:
            if self.is_conv:
                mean = T.mean(input, axis=(0, 2, 3))  # , keepdims=True)
                mean2 = T.mean(input**2, axis=(0, 2, 3))  # , keepdims=True)
            else:
                mean = T.mean(input, axis=0)  # , keepdims=True)
                mean2 = T.mean(input**2, axis=0)  # , keepdims=True)
            std = T.sqrt(T.mean(mean2 - mean**2, axis=0) + self.epsilon)
        return mean, std

    def _output(self, input,  *args, **kwargs):
        '''
        normed=(input-mean)/std
        result=gamma*normed+beta
              =gamma/std*(input-mean)+beta
              =gamma/std*input-gamma/std*mean+beta
              =a*input+b
        a=gamma/std
        b=beta-a*mean
        '''
        if self.affected_channels == self.n_channel:
            affected = input
            unaffected = None
        else:
            affected = input[:, :self.affected_channels]
            unaffected = input[:, self.affected_channels:]
        mean, std = self.getDistribution(affected)

        a = self.gamma / std
        b = self.beta - a * mean
        if self.is_conv:
            a = a.reshape((1, affected.shape[1], 1, 1))
            b = b.reshape((1, affected.shape[1], 1, 1))
        affected = a * affected + b

        if None == unaffected:
            return affected
        else:
            return T.concatenate([affected, unaffected], axis=1)

    def getParams4SL(self, count):
        if None == self.name:
            prefix = 'bn%.3d' % count
            count += 1
        else:
            prefix = self.name
        r = [(prefix + '_m_%d' % self.affected_channels, self.dst_mean),
             (prefix + '_s_%d' % self.affected_channels, self.dst_std), ]
 #       r=[]
        if not self.noparam:
            r += [(prefix + '_r_%d' % self.affected_channels, self.gamma),
                  (prefix + '_b_%d' % self.affected_channels, self.beta), ]
        return r, count

    def weight(self):
        if self.fixed:
            return [] + self.input_layer.weight()
        return [self.gamma] + self.input_layer.weight()

    def bias(self):
        if self.fixed:
            return [] + self.input_layer.bias()
        return [self.beta] + self.input_layer.bias()


class PoolLayer(Layer):

    def __init__(self, input_layer,  pool_size, stride, ):
        #        assert input_shape == input_layer.shape()
        #        self._shape = (input_shape[0], (input_shape[1] - pool_size + stride) / stride, (
        # input_shape[2] - pool_size + stride) / stride, input_shape[3])
        self.input_layer = input_layer
        self.pool_size = (pool_size, pool_size)
        self.stride = (stride, stride)
        self.n_channel = self.input_layer.n_channel

    def _output(self, input,  *args, **kwargs):
        input_shuffled = input
        out = dnn.dnn_pool(input_shuffled,
                           ws=self.pool_size,
                           stride=self.stride)
        return out


class DropoutLayer(Layer):

    def __init__(self, input_layer, dropout):
        self.input_layer = input_layer
        self.dropout = dropout
        self.n_channel = self.input_layer.n_channel
      #  self._shape = input_layer.shape()

    def _output(self, input, dropout_active=True,  *args, **kwargs):
        if not dropout_active:
            return input
        if self.dropout < 0:
            return input

        retain_prob = 1 - self.dropout
        out = input / retain_prob * \
            srng.binomial(
                size=input.shape, p=retain_prob, dtype='int32').astype('float32')
        # apply the input mask and rescale the input accordingly. By doing this
        # it's no longer necessary to rescale the weights at test time.
        return out
#
#
# class BC01toC01BLayer(Layer):
#
#    def _output(self, input,  *args, **kwargs):
# return input.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
#
#
# class C01BtoBC01Layer(Layer):
#
#    def _output(self, input,  *args, **kwargs):
# return input.dimshuffle(3, 0, 1, 2)  # c01b to bc01
#


class FlattenLayer(Layer):

    #    def __init__(self, input_layer, input_shape):
    #        assert input_shape == input_layer.shape()
    #        self.input_layer = input_layer
    #        self._shape = (
    # input_shape[0] * input_shape[1] * input_shape[2], input_shape[3])

    def _output(self, input,  *args, **kwargs):
        # return T.flatten(input.dimshuffle(3, 0, 1, 2), 2)
        return T.flatten(input, 2)  # .dimshuffle(3, 0, 1, 2), 2)


class RandomPixel(Layer):

    def __init__(self, input_layer, batch_size, img_size):
        self.input_layer = input_layer
        self.n_channel = self.input_layer.n_channel

        self.batch_size = batch_size
        self.img_size = img_size

    def _output(self, input,  *args, **kwargs):
        x = srng.uniform(size=(self.batch_size,), high=self.img_size)
        y = srng.uniform(size=(self.batch_size,), high=self.img_size)
        x = T.cast(x, 'int32')
        y = T.cast(y, 'int32')
        r = []
        for i in range(self.batch_size):
            item = input[i]
            item = T.concatenate(
                [item[:, x[i]:, :], item[:, :x[i], :]], axis=1)
            item = T.concatenate(
                [item[:, :, y[i]:], item[:, :, :y[i]]], axis=2)
            r.append(item)
        r = T.stacklists(r)
        return r


class ImageMergeLayer(Layer):

    def __init__(self, input_layer, n_img, n_in, suppression=2):
        self.input_layer = input_layer
        self.n_channel = self.input_layer.n_channel
        self.n_img = n_img
        self.n_in = n_in
        self.suppression = suppression
#        self.W=np.asarray([1]*n_img).astype('float32')
#        self.W=self.W.reshape((n_img,1))

    def _output(self, input, batch_size,  *args, **kwargs):
        n = self.n_img
        n_filters = self.n_in
        out = input.reshape((batch_size, n, n_filters))

#        out=out.dimshuffle(0,2,1)
#        out=T.dot(out,self.W)
#        out=out.reshape((batch_size,n_filters))
#        return out
        # return T.max(out,axis=1)

        out = T.sum(out, axis=1)
        out /= np.float32(n * self.suppression)
        # it works, but i dont know why
        return out
        # out = T.dot(input, self.W.val) + self.b.val
 #       out=T.dot(input,W)
#        input=input.reshape((

 #       return
#    def output(self, *args, **kwargs):
#        self.num_views = num_views
#        self.params = []
#        self.bias_params = []
#        self.my_temp=2*self.num_views
# self.mb_size = self.input_layer.mb_size // self.my_temp # divide by
# total number of parts


#        input_shape = self.input_layer.get_output_shape()
#        input = self.input_layer.output(*args, **kwargs)
# input_r = input.reshape((self.my_temp, self.mb_size, int(np.prod(input_shape[1:])))) # split out the 4* dimension
#        return input_r.transpose(1, 0, 2).reshape(self.get_output_shape())


class ReLULayer(Layer):

    def _output(self, input,  *args, **kwargs):
        return T.maximum(input, 0)


class SigmoidLayer(Layer):

    def _output(self, input,  *args, **kwargs):
        return T.nnet.sigmoid(input)


class LRNLayer(Layer):

    def __init__(self, input_layer, alpha=0.0001, beta=0.75, n=5):
        self.input_layer = input_layer
        self.n_channel = self.input_layer.n_channel
#        self._shape = input_layer.shape()
        self.lrn_func = CrossChannelNormalization(alpha=alpha, beta=beta, n=n)

    def _output(self, input,  *args, **kwargs):
        return self.lrn_func(input)


class DenseLayer(Layer):

    def __init__(self, input_layer, n_in, n_out,
                 weight_np=None, bias_np=None,
                 std=.005, bias=1, fixed=False, name=None, normalize_weight=None):
        #        assert input_shape == input_layer.shape()
        #        self._shape = (n_out, input_shape[1])
        self.input_layer = input_layer
#        n_in = input_shape[0]
        self.n_out = n_out
        self.n_channel = n_out
        self.n_in = n_in

        if None == weight_np:
            self.W = generateWeight((n_in, n_out), std=std)
        else:
            self.W = theano.shared(weight_np)
        if None == bias_np:
            self.b = generateWeight(n_out, init=bias, std=0)
        else:
            self.b = theano.shared(bias_np)
        self.name = name
        self.fixed = fixed
        self.normalize_weight = normalize_weight

    def _output(self, input,  *args, **kwargs):
        # return self.nonlinearity(T.dot(input, self.W) +
        # self.b.dimshuffle('x', 0))
        W = self.W
        if not None == self.normalize_weight:
            epsilon = 1e-6
            W = W * self.normalize_weight / \
                (T.mean(W**2, axis=1)[:, np.newaxis] + epsilon)**.5
        out = T.dot(input, W) + self.b
        return out

    def getParams4SL(self, count):
        if None == self.name:
            w = 'dense%.3d_w' % count
            b = 'dense%.3d_b' % count
            count += 1
        else:
            w = self.name + '_w'
            b = self.name + '_b'
        w += '_%dx%d' % (self.n_in, self.n_out)
        b += '_%dx%d' % (self.n_in, self.n_out)
        return [(w, self.W), (b, self.b)], count

    def weight(self):
        if self.fixed:
            return [] + self.input_layer.weight()
        return [self.W] + self.input_layer.weight()

    def bias(self):
        if self.fixed:
            return [] + self.input_layer.bias()
        return [self.b] + self.input_layer.bias()


class Sparsity(Layer):

    # def getStdAndDensity(self):
    def _output(self, input,  *args, **kwargs):
        input = self.input_layer.output()
        out = T.switch(T.gt(input, 0), 1, 0)
        if out.ndim > 2:
            std = T.std(out, axis=(0, 2, 3))
        else:
            std = T.std(out, axis=0)
        return T.concatenate([T.mean(std).reshape((1,)), T.mean(out).reshape((1,))])
    #    return [T.mean(std), T.mean(out)]


class SoftmaxLayer(Layer):

    def __init__(self, input_layer, y_target):
        self.input_layer = input_layer
        self.y_target = y_target

    def loss(self,  *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        x = T.nnet.softmax(input)
        y = self.y_target
        return -T.mean(T.log(x)[T.arange(y.shape[0]), y])

    def error(self,  *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        x = T.nnet.softmax(input)
        y_pred = T.argmax(x, axis=1)
        y = self.y_target
        return T.mean(T.neq(y_pred, y))

    def _output(self, input,  *args, **kwargs):
        return T.nnet.softmax(input)


class AdditiveLayer(Layer):

    def __init__(self, input_layer, previous_pred, test_pp_only=False):
        self.input_layer = input_layer
        self.previous_pred = previous_pred
        self.test_pp_only = test_pp_only

    def _output(self, input,  *args, **kwargs):
        if self.test_pp_only:
            return input * 0 + self.previous_pred
        return input + self.previous_pred


class MeanSquaredErrorLayer(Layer):

    def __init__(self,  input_layer, y_target):  # , rank=5):
        self.input_layer = input_layer
        self.y_target = y_target

    def loss(self,  *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        error = input - self.y_target
        error = T.mean(error**2)
        return error


class OrdSoftmaxLayer(Layer):

    def __init__(self,  input_layer, y_target, y_prediction=None, rank=5):
        self.rank = rank
        self.input_layer = input_layer
        self.y_target = y_target
        self.y_prediction = y_prediction

    def _softmax(self,  *args, **kwargs):
        input = self.input_layer.output(*args, **kwargs)
        r = [T.nnet.softmax(input[:, 2 * i:2 * (i + 1)])
             for i in range(self.rank - 1)]
        return r

    def loss(self,  *args, **kwargs):
        y = self.y_target
        batch_size = y.shape[0]
        r = self._softmax(*args, **kwargs)
        r = [T.log(r[i])[T.arange(batch_size), T.gt(y, i)]
             for i in range(self.rank - 1)]
        r = T.concatenate(r, axis=0)
        return -T.mean(r)

    def mse(self,  *args, **kwargs):
        r = self._softmax(*args, **kwargs)
        s = T.concatenate(r, axis=1)
        error = s - self.y_prediction[:, :(self.rank - 1) * 2]
        error = T.mean(error**2)
        return error

    def _output(self, input,  *args, **kwargs):
        b, c = input.shape
        return T.nnet.softmax(input)
# class Weight(object):
#
#    def __init__(self, w_shape, mean=0, std=0.01):
#        super(Weight, self).__init__()
#        if std != 0:
#            self.np_values = np.asarray(
#                rng.normal(mean, std, w_shape), dtype=theano.config.floatX)
#        else:
#            self.np_values = np.cast[theano.config.floatX](
#                mean * np.ones(w_shape, dtype=theano.config.floatX))
#
#        self.val = theano.shared(value=self.np_values)
#
#    def save_weight(self, dir, name):
#        print 'weight saved: ' + name
#        np.save(dir + name + '.npy', self.val.get_value())
#
#    def load_weight(self, dir, name):
#        print 'weight loaded: ' + name
#        self.np_values = np.load(dir + name + '.npy')
#        self.val.set_value(self.np_values)
#


def generateWeight(w_shape,  std, init=0):
    if std != 0:
        np_values = np.asarray(
            rng.normal(init, std, w_shape), dtype=theano.config.floatX)
    else:
        np_values = np.cast[theano.config.floatX](
            init * np.ones(w_shape, dtype=theano.config.floatX))
    return theano.shared(np_values)


def saveWeights(params, prefix):
    i = 0
    for param in params:
        i += 1
        path = prefix + '-%.2d.npy' % i
        parent = os.path.split(path)[0]
        if not os.path.exists(parent):
            os.makedirs(parent)
        np.save(path, param.get_value())


def loadWeights(params, prefix):
    i = 0
    for param in params:
        i += 1
        path = prefix + '-%.2d.npy' % i
        data = np.load(path)
        param.set_value(data)
