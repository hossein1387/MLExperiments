import time

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def clip(x):
    return T.clip((x+1.)/2.,0,1)

def hard_sigmoid(x):
    return np.clip((x+1.)/2.,0,1)
#    return np.maximum(0.0, np.minimum(1.0, (x+1.0)/2.0))

def binarize(W):
    Wb = hard_sigmoid(W)
#    import ipdb; ipdb.set_trace()  # <--- *BAMF!*
    Wb = np.random.binomial(n=1, p=Wb, size=np.shape(Wb))
    Wb.astype(float)
    Wb = np.where(Wb==1.0,1,-1)
    Wb.astype(float)
    return Wb

def binarize_theano(W):
    Wb = clip(W)
    Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)
    Wb = T.cast(T.switch(Wb,1,-1), theano.config.floatX)
    return Wb

srng =  RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
data_img = np.load('train_images.npy').reshape((600, 100, 1, 28, 28))
data_tar = np.load('train_labels.npy').reshape((600, 100)).astype('int64')

# The following symbols are defined as a placeholder so that later we can 
# fed the real data to them
x = T.tensor4('x')
y = T.lvector('y')

# The following variables are defined as shared. Since they will have initial
# values.
# Lets use 5*5 filter
w_1 = theano.shared((np.random.rand(20, 1, 5, 5) - 0.5) * 2 * np.sqrt(6 / (45.0)), 'w_1')
b_1 = theano.shared(np.zeros(20), 'b_1')

w_2 = theano.shared((np.random.rand(50, 20, 5, 5) - 0.5) * 2 * np.sqrt(6 / (550.0)), 'w_2')
b_2 = theano.shared(np.zeros(50), 'b_2')

w_3 = theano.shared((np.random.rand(4 * 4 * 50, 500) - 0.5) * 2 * np.sqrt(6 / (1300.0)), 'w_3')
b_3 = theano.shared(np.zeros(500), 'b_3')

w_4 = theano.shared((np.random.randn(500, 10) - 0.5) * 2 * np.sqrt(6 / (510.0)), 'w_4')
b_4 = theano.shared(np.zeros(10), 'b_4')

# Computational Graph
w1_b = binarize_theano(w_1)
pa_1 = T.nnet.conv2d(x, w1_b) + b_1.dimshuffle('x', 0, 'x', 'x')
a_1 = pool.pool_2d(T.tanh(pa_1), (2, 2), ignore_border=True)

w2_b = binarize_theano(w_2)
pa_2 = T.nnet.conv2d(a_1, w2_b) + b_2.dimshuffle('x', 0, 'x', 'x')
a_2 = pool.pool_2d(T.tanh(pa_2), (2, 2), ignore_border=True)

w3_b = binarize_theano(w_3)
pa_3 = T.dot(a_2.flatten(2), w3_b) + b_3.dimshuffle('x', 0)
a_3 = T.tanh(pa_3)

w4_b = binarize_theano(w_4)
pa_4 = T.dot(a_3, w4_b) + b_4.dimshuffle('x', 0)
a_4 = T.nnet.softmax(pa_4)
y_hat = a_4

cost = T.mean(T.nnet.categorical_crossentropy(y_hat, y))
cost.name = 'CE'
params_b = [w1_b, b_1, w2_b , b_2, w3_b, b_3, w4_b, b_4]
params  = [w_1, b_1, w_2 , b_2, w_3, b_3, w_4, b_4]
dparams = T.grad(cost, params_b)
#dparams = T.grad(cost, params)

alpha = 0.15
updates = []

for i, p, dp in zip(range(len(params)), params, dparams):
    p_val = p.get_value()
    if len(p_val.shape) > 1:
        updates += [(p, clip((p - alpha * dp)))]
    else:
        updates += [(p, p - alpha * dp)]

# Defining inputs and outputs
f_eval = theano.function([x], y_hat)
f_train = theano.function([x, y], cost, updates=updates)
f_train = theano.function([x, y], [cost, w_1], updates=updates)
f_test = theano.function([], [w_1, binarize_theano(w_1)])


for e in range(20):
    batch_errs = []
    for i in range(10):
        batch_img = data_img[i]
        batch_tar = data_tar[i]
        pred = np.argmax(f_eval(batch_img), axis=1)
        batch_err = (batch_tar == pred).astype('int').sum() / 100.0
        batch_errs += [batch_err]
    print 'error: ' + str((1 - np.mean(batch_errs)) * 100)
    for i in range(10):
        batch_img = data_img[i]
        batch_tar = data_tar[i]
        #import ipdb; ipdb.set_trace()  # <--- *BAMF!*
        cost = f_train(batch_img, batch_tar)
        w1, w1_b = f_test()
        #print w1_b
        #import ipdb; ipdb.set_trace()  # <--- *BAMF!*
