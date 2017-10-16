import time

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import pdb

##################################################################################
# Globals
##################################################################################
srng =  RandomStreams(lasagne.random.get_rng().randint(1, 214))
data_img = np.load('train_images.npy').reshape((600, 100, 1, 28, 28)) # input dataset
data_tar = np.load('train_labels.npy').reshape((600, 100)).astype('int64')
alpha = 0.001
epsilon = 1e-6
##################################################################################
# Utility Functions
##################################################################################
def clip(x):
    return T.clip((x+1.)/2.,0,1)

def binarize_theano(W):
    Wb = clip(W)
    Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)
    Wb = T.cast(T.switch(Wb,1,-1), theano.config.floatX)
    return Wb

##################################################################################
# Theano symbols definition
##################################################################################
# The following symbols are defined as a placeholder so that later we can 
# fed the real data to them
x = T.tensor4('x')
y = T.lvector('y')

# The following variables are defined as shared. Since they will have initial
# values.
# Lets use 5*5 filter
w_1   = theano.shared((np.random.rand(20, 1, 5, 5) - 0.5) * 2 * np.sqrt(1.5 / (45.0)), 'w_1')
gama_1= theano.shared(np.ones(20), 'gama_1')
b_1   = theano.shared(np.zeros(20), 'b_1')

w_2   = theano.shared((np.random.rand(50, 20, 5, 5) - 0.5) * 2 * np.sqrt(1.5 / (550.0)), 'w_2')
gama_2 = theano.shared(np.ones(50), 'gama_2')
b_2   = theano.shared(np.zeros(50), 'b_2')

w_3   = theano.shared((np.random.rand(4 * 4 * 50, 500) - 0.5) * 2 * np.sqrt(1.5 / (1300.0)), 'w_3')
gama_3= theano.shared(np.ones(500), 'gama_3')
b_3   = theano.shared(np.zeros(500), 'b_3')

w_4   = theano.shared((np.random.randn(500, 10) - 0.5) * 2 * np.sqrt(1.5 / (510.0)), 'w_4')
gama_4= theano.shared(np.ones(10), 'gama_4')
b_4   = theano.shared(np.zeros(10), 'b_4')

##################################################################################
# Computational Graph
##################################################################################
w1_b    = binarize_theano(w_1)
pa_1    = T.nnet.conv2d(x, w1_b) 
mu_1    = T.mean(pa_1, axis=0, keepdims=True)
std_1   = T.sqrt(T.var(pa_1, axis=0, keepdims=True) + 1e-6) #T.std didn't work by it self on this layer, std was too small
bn_1    = ((pa_1 - mu_1)/(std_1+epsilon))*gama_1.dimshuffle('x', 0, 'x', 'x') + b_1.dimshuffle('x', 0, 'x', 'x')
a_1     = pool.pool_2d(T.tanh(bn_1), (2, 2), ignore_border=True)

w2_b    = binarize_theano(w_2)
pa_2    = T.nnet.conv2d(a_1, w2_b)
mu_2    = T.mean(pa_2, axis=0, keepdims=True)
std_2   = T.std(pa_2, axis=0, keepdims=True)
bn_2    = ((pa_2 - mu_2)/(std_2+epsilon))*gama_2.dimshuffle('x', 0, 'x', 'x') + b_2.dimshuffle('x', 0, 'x', 'x')
a_2     = pool.pool_2d(T.tanh(bn_2), (2, 2), ignore_border=True)

w3_b    = binarize_theano(w_3)
pa_3    = T.dot(a_2.flatten(2), w3_b)
mu_3    = T.mean(pa_3, axis=0, keepdims=True)
std_3   = T.std(pa_3, axis=0, keepdims=True)
bn_3    = ((pa_3 - mu_3)/(std_3+epsilon))*gama_3.dimshuffle('x', 0) + b_3.dimshuffle('x', 0)
a_3     = T.tanh(bn_3)

w4_b    = binarize_theano(w_4)
pa_4    = T.dot(a_3, w4_b)
mu_4    = T.mean(pa_4, axis=0, keepdims=True)
std_4   = T.std(pa_4, axis=0, keepdims=True)
bn_4    = ((pa_4 - mu_4)/(std_4+epsilon))*gama_4.dimshuffle('x', 0) + b_4.dimshuffle('x', 0)
a_4     = T.nnet.softmax(bn_4)
y_hat   = a_4

##################################################################################
# Parameter updates
##################################################################################
cost = T.mean(T.nnet.categorical_crossentropy(y_hat, y))
cost.name = 'CE'
params_b  = [ gama_1, w1_b, b_1, gama_2, w2_b , b_2, gama_3, w3_b, b_3, gama_4, w4_b, b_4] # Binary weights
params    = [ gama_1, w_1,  b_1, gama_2, w_2 ,  b_2, gama_3, w_3,  b_3, gama_4, w_4,  b_4] # Float weights
# Computing gradients with regards to binary weights
dparams   = T.grad(cost, params_b)
updates   = []

for i, p, dp in zip(range(len(params)), params, dparams):
    p_val = p.get_value()   
    if len(p_val.shape) > 1:
        updates += [(p, clip(p - alpha * dp))]
    else:
        updates += [(p, p - alpha * dp)]

##################################################################################
# Defining inputs and outputs
##################################################################################
f_eval  = theano.function([x], y_hat)
f_train = theano.function([x, y], [pa_1, cost], updates=updates)
f_pa_1  = theano.function([x], [T.sqrt(T.sum(std_1 ** 2)), T.sqrt(T.sum(bn_1**2))])
##################################################################################
# Training and computing error
##################################################################################

for e in range(20):
    batch_errs = []
    for i in range(10):
        batch_img = data_img[i] / 255.0
        batch_tar = data_tar[i]
        pred = np.argmax(f_eval(batch_img), axis=1)
        batch_err = (batch_tar == pred).astype('int').sum() / 100.0
        batch_errs += [batch_err]
        # import pdb; pdb.set_trace()
    print 'error: ' + str((1 - np.mean(batch_errs)) * 100)
    for i in range(10):
        batch_img = data_img[i] / 255.0
        batch_tar = data_tar[i]
        pa, cost  = f_train(batch_img, batch_tar)
        print w_4.eval()
