import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import scale
from numpy import pi, sqrt
import math
import random

from matplotlib import pyplot as plt

eng, q = sf.Engine(1)


iterations = 45
shots = 10
parabolic_min = 1
possible_parameters = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
testing_trials = 100
optimal_value = math.inf
simulation = []

#Calculating loss

def function_optimize(x, parabolic_min):
    y = (x - parabolic_min)**2
    return y

#Run the photonic quantum circuit

def run_circuit(alpha_param, beta_param, parabolic_min):

    with eng:

        Squeezed(-0.5,0) | q[0]

        Zgate(parabolic_min*4*beta_param[0]) | q[0]
        Pgate(-4*beta_param[0]) | q[0]
        Rgate(-1*alpha_param[0]) | q[0]
        Zgate(parabolic_min*4*beta_param[1]) | q[0]
        Pgate(-4*beta_param[1]) | q[0]
        Rgate(-1*alpha_param[1]) | q[0]
        Zgate(parabolic_min*4*beta_param[2]) | q[0]
        Pgate(-4*beta_param[2]) | q[0]
        Rgate(-1*alpha_param[2]) | q[0]

        MeasureX | q[0]

        state = eng.run('gaussian', cutoff_dim=5)

    return q[0].val

#Search for the optimal value

op = 0

alpha_param = [0.5, 0.5, 0.5]
beta_param = [0.5, 0.5, 0.5]

for i in range(0, iterations):
    result = 0
    grid = []
    for h in range(0, shots):
        hello = run_circuit(alpha_param, beta_param, parabolic_min)
        result = result+hello
        grid.append(hello)
    result = result/shots
    calculation = function_optimize(result, parabolic_min)

    yer_a = alpha_param
    yer_b = beta_param

    if (calculation < optimal_value):
        simulation = grid
        optimal_value = calculation
        the_x_measurement = result
        op = [[alpha_param[0], alpha_param[1], alpha_param[2]], [beta_param[0], beta_param[1], beta_param[2]]]

    alpha_param[0] = possible_parameters[random.randint(0, len(possible_parameters)-1)]
    alpha_param[1] = possible_parameters[random.randint(0, len(possible_parameters)-1)]
    alpha_param[2] = possible_parameters[random.randint(0, len(possible_parameters)-1)]

    beta_param[0] = possible_parameters[random.randint(0, len(possible_parameters)-1)]
    beta_param[1] = possible_parameters[random.randint(0, len(possible_parameters)-1)]
    beta_param[2] = possible_parameters[random.randint(0, len(possible_parameters)-1)]


optimal_parameters = op

print(the_x_measurement)
print(optimal_parameters)


a = -1*optimal_parameters[0][0]/(2*math.pi)
b = 4*parabolic_min*optimal_parameters[1][0]
c = -4*optimal_parameters[1][0]
d = -1*optimal_parameters[0][1]/(2*math.pi)
e = 4*parabolic_min*optimal_parameters[1][1]
f = -4*optimal_parameters[1][1]
g = -1*optimal_parameters[0][2]/(2*math.pi)
h = 4*parabolic_min*optimal_parameters[1][2]
i = -4*optimal_parameters[1][2]
print([b, c, a, e, f, d, h, i, g])


x_arr = range(0, testing_trials)
y_arr = []

for i in range(0, testing_trials):

    state = run_circuit(optimal_parameters[0], optimal_parameters[1], parabolic_min)
    y_arr.append(state)

'''
plt.scatter(x_arr, y_arr)
plt.plot(x_arr, other_y_arr)
#plt.show()
'''

print(y_arr)
print(sum(y_arr)/len(y_arr))
