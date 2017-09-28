import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

steps = 0.01
def get_dist(xin, steps):
    cur_bnd = 0
    iter = 0
    x_rand_dist = []
    for i in np.arange(0, 1, steps):
        xin_cpy = xin
        x_rand_dist_indx = np.where((xin_cpy>=cur_bnd))
        xin_cpy = xin[x_rand_dist_indx]
        x_rand_dist_indx = np.where((xin_cpy<cur_bnd+steps))
        x_rand_dist.append(np.shape(x_rand_dist_indx)[1])
        cur_bnd += steps
        iter += 1
    return x_rand_dist

x_rand  = np.random.rand(100000)
x_randn = np.random.randn(100000)

x = np.arange(0, 1, steps)
y_rand = get_dist(x_rand, steps)
y_randn = get_dist(x_randn, steps)

plt.bar(x, y_rand, align='center', alpha=0.5, width=0.01)
plt.show()

