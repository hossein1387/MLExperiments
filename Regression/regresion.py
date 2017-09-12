import numpy as np
import matplotlib.pyplot as plt

# Generating data
m = np.arange(-5,5, 0.1)
n = -3*m + 2 + 2*np.random.rand(len(m))

# Creating our model
a = 1
b = 0
x = np.arange(-5,5, 0.1)
y = a*x + b
thr = 0.001
prev_cost = 10000
cost = 0
iter=0

while (np.abs(cost-prev_cost) >= 0.001):
    y = a*x + b
    prev_cost = cost
    cost = np.sum(np.abs(y-n))
    da = np.sum(np.sign(y-n)*x)
    db = np.sum(np.sign(y-n))
    a = a - 0.001*da
    b = b - 0.001*db
    iter += 1
    print "iter=", iter, " cost=", cost

print "got answer after ", iter, "iterations.\n", a, "x + ", b, " with cost=", cost
plt.scatter(m, n)
plt.plot(x, y, color='r')
plt.show()
