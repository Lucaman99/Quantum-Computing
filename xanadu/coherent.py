import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
from matplotlib import pyplot as plt
from scipy import optimize

data_set = []

y_values = [0, 0, 0, 0, 0]

x_values = [0, 1, 2, 3, 4]

all_x  = [x_values[0]]

def run():
    eng, q = sf.Engine(4, hbar=0.5)

    with eng:
        Coherent(1+0j) | q[0]
        Measure | q[0]

    state = eng.run('fock', cutoff_dim=5)

    return q[0]

for i in range (0, 5000):
    data_set.append(run().val)

for k in range (0, len(data_set)):
    if (int(data_set[k]) < 5):
        y_values[int(data_set[k])] = y_values[int(data_set[k])] + 1

all_y = [y_values[0]]

for l in range (1, len(y_values)):
    dif = y_values[l-1]-y_values[l]
    for o in range(1, 11):
        all_x.append(x_values[l-1]+(o/10))
        all_y.append(dif*(1-o/10)+y_values[l])

print(all_x)
print(all_y)


def input_function(x, a, b, c, d):
    return a*np.exp(-1 * b*(x+c)**2) + d


params, params_covariance = optimize.curve_fit(input_function, all_x, all_y)

print(params)

plt.plot(all_x, input_function(all_x, params[0], params[1], params[2], params[3]),
         label='Fitted function', zorder=5)

plt.title('Graph 1')
plt.scatter(all_x, all_y, zorder=10)
plt.bar(x_values, y_values, zorder=0)
plt.show()
