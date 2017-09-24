import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from PIL import Image
import ipdb;

data_img = np.load('train_images.npy').reshape((600, 100, 1, 28, 28))
data_tar = np.load('train_labels.npy').reshape((600, 100)).astype('int64')
# plt.imshow(data_img[0, 2].reshape(28, 28), cmap='gray')
# plt.show()

# The following symbols are defined as a placeholder so that later we can 
# fed the real data to them
x = T.tensor4('x')
y = T.lvector('y')

# The following variables are defined as shared. Since they will have initial
# values.
# Lets use 5*5 filter
w_1 = theano.shared(0.05 * np.random.randn(8, 1, 5, 5), 'w_1')
b_1 = theano.shared(np.zeros(8), 'b_1')

w_2 = theano.shared(0.05 * np.random.randn(8, 8, 5, 5), 'w_2')
b_2 = theano.shared(np.zeros(8), 'b_2')

w_3 = theano.shared(0.05 * np.random.randn(16, 8, 5, 5), 'w_3')
b_3 = theano.shared(np.zeros(16), 'b_3')

w_4 = theano.shared(0.05 * np.random.randn(16, 16, 5, 5), 'w_4')
b_4 = theano.shared(np.zeros(16), 'b_4')

w_5 = theano.shared(0.05 * np.random.randn(2304, 1000), 'w_5')
b_5 = theano.shared(np.zeros(1000), 'b_5')

w_6 = theano.shared(0.05 * np.random.randn(1000, 100), 'w_6')
b_6 = theano.shared(np.zeros(100), 'b_6')

w_7 = theano.shared(0.05 * np.random.randn(100, 10), 'w_7')
b_7 = theano.shared(np.zeros(10), 'b_7')

# Computational Graph
pa_1 = T.nnet.conv2d(x, w_1) + b_1.dimshuffle('x', 0, 'x', 'x')
a_1 = T.maximum(0, pa_1)

pa_2 = T.nnet.conv2d(a_1, w_2) + b_2.dimshuffle('x', 0, 'x', 'x')
a_2 = T.maximum(0, pa_2)

pa_3 = T.nnet.conv2d(a_2, w_3) + b_3.dimshuffle('x', 0, 'x', 'x')
a_3 = T.maximum(0, pa_3)

pa_4 = T.nnet.conv2d(a_3, w_4) + b_4.dimshuffle('x', 0, 'x', 'x')
a_4 = T.maximum(0, pa_4)

pa_5 = T.dot(a_4.flatten(2), w_5) + b_5.dimshuffle('x', 0)
a_5 = T.maximum(0, pa_5)

pa_6 = T.dot(a_5, w_6) + b_6.dimshuffle('x', 0)
a_6 = T.maximum(0, pa_6)

pa_7 = T.dot(a_6, w_7) + b_7.dimshuffle('x', 0)
a_7 = T.nnet.softmax(pa_7)
y_hat = a_7

cost = T.mean(T.nnet.categorical_crossentropy(y_hat, y))
cost.name = 'CE'
dw_1, db_1, dw_2, db_2, dw_3, db_3, dw_4, db_4, dw_5, db_5, dw_6, db_6, dw_7, db_7 = T.grad(cost, [w_1, b_1, w_2, b_2, w_3, b_3, w_4, b_4, w_5, b_5, w_6, b_6, w_7, b_7])

alpha = 0.1
updates = [(w_1, w_1 - alpha * dw_1),
           (b_1, b_1 - alpha * db_1),
           (w_2, w_2 - alpha * dw_2),
           (b_2, b_2 - alpha * db_2),
           (w_3, w_3 - alpha * dw_3),
           (b_3, b_3 - alpha * db_3),
           (w_4, w_4 - alpha * dw_4),
           (b_4, b_4 - alpha * db_4),
           (w_5, w_5 - alpha * dw_5),
           (b_5, b_5 - alpha * db_5),
           (w_6, w_6 - alpha * dw_6),
           (b_6, b_6 - alpha * db_6),
           (w_7, w_7 - alpha * dw_7),
           (b_7, b_7 - alpha * db_7)]

# Defining inputs and outputs
f_eval = theano.function([x], y_hat)
f_train = theano.function([x, y], cost, updates=updates)

batch_errs = []
iter = 0
cost_per_iter = []
err_per_iter = []

for e in range(10):
    for i in range(600):
        batch_img = data_img[i]
        batch_tar = data_tar[i]
        pred = np.argmax(f_eval(batch_img), axis=1)
        batch_err = (batch_tar == pred).astype('int').sum() / 100.0
        batch_errs += [batch_err]
        err_per_iter.append((1 - np.mean(batch_errs)) * 100)
    print 'error: ' + str((1 - np.mean(batch_errs)) * 100)
    for i in range(600):
        batch_img = data_img[i]
        batch_tar = data_tar[i]
        cost = f_train(batch_img, batch_tar)
        cost_per_iter.append(cost)
        iter += 1
    print 'cost: ' + str(cost) + '\n'
