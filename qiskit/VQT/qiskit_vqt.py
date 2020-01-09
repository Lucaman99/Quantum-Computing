#Importing the necessary dependencies

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
import math
import random
import numpy as np
from scipy.optimize import minimize
import math
import scipy
from matplotlib import pyplot as plt

# Sets the values of the initial parameters

beta = 1 #Note that B = 1/T
qubit = 3
depth = 1
trotter = 1
dimension = qubit

double_pair = [[[0, 1], [1, 2]]]
single_pair = [[0, 1, 2]]

#Creates the RYY gate using the RXX gate and single-qubit rotations

def ryy_gate(circ, qubit1, qubit2, theta):

    circ.s(qubit1)
    circ.s(qubit2)
    circ.rxx(theta, qubit1, qubit2)
    circ.sdg(qubit1)
    circ.sdg(qubit2)

#Creates the double rotational ansatz outlined in the paper (modified slightly for the 1D Heisenberg model)

def double_rotation(circ, trot_depth, phi_params, qubit1, qubit2):
        
    circ.rzz(phi_params[0], qubit1, qubit2)
    circ.rxx(phi_params[1], qubit1, qubit2)
    ryy_gate(circ, qubit1, qubit2, phi_params[2])

# Creates the single rotational ansatz outlined in the paper (modified slightly for the 1D Heisenberg model)

def single_rotation(circ, trot_depth, phi_params, qubit):

    circ.rz(phi_params[0], qubit)
    circ.ry(phi_params[1], qubit)
    circ.rx(phi_params[2], qubit)

# Creates the probability distribution according to the theta parameters

def create_dist(theta_param):

    prob = []
    for i in range(0, len(theta_param)):
        prob.append([math.exp(-1*theta_param[i]), 1-math.exp(-1*theta_param[i])])

    return prob

#Creates the initialization unitary for each of the computational basis states

def create_v_gate(circ, prep_state):

    for i in range(0, len(prep_state)):
        if (prep_state[i] == 1):
            circ.x(i)

#Prepares the Hamiltonian matrix for the X-Hamiltonian

def create_hamiltonian_matrix(n):

    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    identity = np.array([[1, 0], [0, 1]])
    
    matrix = np.zeros((2**n, 2**n))

    for i in range(0, n-1):
        m = 1
        for j in range(0, n):
            if (j == i or j == i+1):
                m = np.kron(m, pauli_x)
            else:
                m = np.kron(m, identity)
        matrix = np.add(matrix, m)
    
    for i in range(0, n-1):
        m = 1
        for j in range(0, n):
            if (j == i or j == i+1):
                m = np.kron(m, pauli_y)
            else:
                m = np.kron(m, identity)
        matrix = np.add(matrix, m)
    
    for i in range(0, n-1):
        m = 1
        for j in range(0, n):
            if (j == i or j == i+1):
                m = np.kron(m, pauli_z)
            else:
                m = np.kron(m, identity)
        matrix = np.add(matrix, m)
    
    return matrix

# Creates the U gate unitary to apply to the initial qubit state

def add_u_gate(circ, double_phi, single_phi, qubit):
    for j in range(0, depth):
        
        for i in range(0, qubit-1):
            double_rotation(circ, trotter, double_phi[j][i], double_pair[j][i][0], double_pair[j][i][1])
            
        for i in range(0, qubit):
            single_rotation(circ, trotter, single_phi[j][i], i)
            
        for i in range(0, qubit-1):
            double_rotation(circ, trotter, double_phi[j][i], double_pair[j][i][0], double_pair[j][i][1])


#Numerical cost function

def cost_function(param):
    
    ham_matrix = create_hamiltonian_matrix(qubit)
    
    double_phi = [[param[0:3], param[3:6]]]
    single_phi = [[param[6:9], param[9:12], param[12:15]]]
    theta = param[15:18]

    opt_prob_dist = create_dist(theta)

    p = []
    for i in range(0, 2):
        for j in range(0, 2):
            for v in range(0, 2):
                p.append(opt_prob_dist[0][i]*opt_prob_dist[1][j]*opt_prob_dist[2][v])

    state = np.zeros((2**dimension, 2**dimension))
    
    for l in range(0, 2**dimension):

        circ = QuantumCircuit(dimension)

        create_v_gate(circ, [int(i) for i in list(bin(l)[2:].zfill(dimension))])
        add_u_gate(circ, double_phi, single_phi, qubit)
        
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend)
        result = job.result()
        outputstate = result.get_statevector(circ)
        
        state = np.add(state, p[l]*np.outer(outputstate, np.conj(outputstate)))
    
    entropy = -1*np.trace(np.matmul(state, scipy.linalg.logm(state)))
    
    cost = beta*np.trace(np.matmul(ham_matrix, state)) - entropy
    print(cost.real)
    return cost.real


#Creates the density matrix visualizer

def create_density_plot(data, re):

    array = np.array(data)
    plt.matshow(array)
    plt.colorbar()
    if (re == 1):
        plt.title("Learned State")
    if (re == 2):
        plt.title("Target State")
    plt.show()


#Creates lits of random numbers as the initial parameters fed into the circuit

init1 = [random.randint(100, 200)/100 for i in range(0, qubit)]
init2 = [random.randint(-100, 100)/50 for i in range(0, 5*qubit)]
init = init2+init1

#Creates the optimization process

out = minimize(cost_function, x0=init, method="COBYLA", options={'maxiter':2000})
g = out['x']
print(out)

param = g

# Prepares the learned state

double_phi = [[param[0:3], param[3:6]]]
single_phi = [[param[6:9], param[9:12], param[12:15]]]
theta = param[15:18]

opt_prob_dist = create_dist(theta)

p = []
for i in range(0, 2):
    for j in range(0, 2):
        for v in range(0, 2):
            p.append(opt_prob_dist[0][i]*opt_prob_dist[1][j]*opt_prob_dist[2][v])
                              
state = np.zeros((2**dimension, 2**dimension))

for l in range(0, 2**dimension):
    
    circ = QuantumCircuit(dimension, dimension)
    create_v_gate(circ, [int(i) for i in list(bin(l)[2:].zfill(dimension))])
    add_u_gate(circ, double_phi, single_phi, qubit)
    
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend)
    result = job.result()
    outputstate = result.get_statevector(circ)
    
    state = np.add(state, p[l]*np.outer(outputstate, np.conj(outputstate)))
    
ham_matrix = create_hamiltonian_matrix(qubit)
entropy = -1*np.trace(np.matmul(state, scipy.linalg.logm(state)))
ev = np.trace(np.matmul(ham_matrix, state))

create_density_plot(state.real, 1)

print("Final Entropy: "+str(entropy.real))
print("Final Expectation Value: "+str(ev.real))
print("Final Cost: "+str(beta*ev.real - entropy.real))


# Prepares the target state (for comparison0


h = create_hamiltonian_matrix(dimension)
ya = -1*float(beta)*h
new_matrix = scipy.linalg.expm(np.array(ya))
norm = np.trace(new_matrix)
final_target = (1/norm)*new_matrix

entropy = -1*np.trace(np.matmul(final_target, scipy.linalg.logm(final_target)))

create_density_plot(final_target.real, 2)

print("Entropy: "+str(entropy))
print("Expectation Value: "+str(np.trace(np.matmul(final_target, h))))

real_cost = beta*np.trace(np.matmul(final_target, h)) - entropy
print("Final Cost: "+str(real_cost))
