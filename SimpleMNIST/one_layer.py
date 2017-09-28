import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from PIL import Image

data_img = np.load('train_images.npy').reshape((600, 100, 784))
data_tar = np.load('train_labels.npy').reshape((600, 100)).astype('int64')
# plt.imshow(data_img[0, 2].reshape(28, 28), cmap='gray')
# plt.show()

# The following symbols are defined as a placeholder so that later we can 
# fed the real data to them
x = T.matrix('x')
y = T.lvector('y')

# The following variables are defined as shared. Since they will have initial
# values.
w_1 = theano.shared(0.05 * np.random.randn(784, 500), 'w_1')
b_1 = theano.shared(np.zeros(500), 'b_1')
# Computational Graph
pa_1 = T.dot(x, w_1) + b_1.dimshuffle('x', 0)
a_1 = T.nnet.sigmoid(pa_1)
y_hat = a_1

cost = T.mean(T.nnet.categorical_crossentropy(y_hat, y))
cost.name = 'CE'
dw_1, db_1 = T.grad(cost, [w_1, b_1])

alpha = 0.01
updates = [(w_1, w_1 - alpha * dw_1),
           (b_1, b_1 - alpha * db_1)]

# Defining inputs and outputs
f_eval = theano.function([x], y_hat)
f_train = theano.function([x, y], cost, updates=updates)

batch_errs = []
iter = 0
cost_per_iter = []
err_per_iter = []

for e in range(30):
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

x = np.arange(0,iter, 1)
plt.plot(x, cost_per_iter, color='b', label='cost')
plt.plot(x, err_per_iter, color='r', label='error')
plt.legend(loc='upper right', shadow=True)
plt.show()
#import ipdb; ipdb.set_trace()
