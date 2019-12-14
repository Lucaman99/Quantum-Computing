# Importing all of the necessary dependencies

import cirq
import random
import numpy as np
import math
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# Note: Beta is defined as inverse temperature, B = 1/T

beta = 0.5
transverse_field_strength = 1
testing_trials = 100
qubit_number = 3
depth = 1

# Initializing the qubits

qubits_a = []
qubits_b = []

qubits = []

for i in range(0, qubit_number):
    qubits_a.append(cirq.GridQubit(1, i))
    qubits_b.append(cirq.GridQubit(2, i))

    qubits.append(cirq.GridQubit(1, i))
    qubits.append(cirq.GridQubit(2, i))

# Defining the cost Hamiltonian

def create_cost_ham(qubits_x, qubit_number, parameter_list):

    # We'll start by experimenting with a simple Ising model. First, we apply the transverse field

    for i in range(0, qubit_number):
        yield cirq.Rz(-2*transverse_field_strength*parameter_list[0]).on(qubits_x[i])

    # We can now apply neighbour interactions between qubits

    for i in range(0, qubit_number-1):
        yield cirq.XXPowGate(exponent= 2*parameter_list[1]/math.pi, global_shift=-0.5).on(qubits_x[i], qubits_x[i+1])

# Calculates the cost Hamiltonian matrix for the Ising model

def calculate_ising_matrix(qubit_number, transverse_field_strength):

    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    identity = np.array([[1, 0], [0, 1]])

    # Creates the ineraction term of the matrix

    total_x_matrix = np.zeros((2**qubit_number, 2**qubit_number))

    for i in range(0, qubit_number-1):
        matrix = 1
        for k in range(0, qubit_number):
            if (k == i or k == i+1):
                matrix = np.kron(matrix, pauli_x)
            else:
                matrix = np.kron(matrix, identity)

        total_x_matrix = np.add(total_x_matrix, matrix)

    # Creates the transverse field component of the matrix

    total_z_matrix = np.zeros((2**qubit_number, 2**qubit_number))

    for i in range(0, qubit_number):
        matrix = 1
        for k in range(0, qubit_number):
            if (k == i):
                matrix = np.kron(matrix, pauli_z)
            else:
                matrix = np.kron(matrix, identity)
        total_z_matrix = np.add(total_z_matrix, matrix)

    final_matrix = np.add(total_x_matrix, transverse_field_strength*total_z_matrix)
    return final_matrix

# Calculating the eigenvalues and eigenvectors of the cost Hamiltonian

def find_eigenvec_eigenval(matrix):

    value, vector = np.linalg.eig(matrix)
    return [value, vector]

# Preaparing the partition function and each of the probability amplitudes of the diifferent terms in the TFD state

def calculate_terms_partition(eigenvalues):

    list_terms = []
    partition_sum = 0
    for i in eigenvalues:
        list_terms.append(math.exp(-0.5*beta*i))
        partition_sum = partition_sum + math.exp(-1*beta*i)

    return [list_terms, math.sqrt(partition_sum)]

# Preparing the initial, maximally entangled state of the qubits

def prepare_entangled_states(qubits_a, qubits_b, qubit_number):

    for i in range(0, qubit_number):

        yield cirq.H.on(qubits_a[i])
        yield cirq.CNOT.on(qubits_a[i], qubits_b[i])
        #yield cirq.X.on(qubits_b[i])
        #yield cirq.Z.on(qubits_a[i])

# Defining the interaction-mixer Hamiltonian

def create_mixer_ham(qubits_a, qubits_b, qubit_number, parameter_list):

    # Implements the exp(ZZ) operation on all entangled states
    for i in range(0, qubit_number):
        yield cirq.ZZPowGate(exponent= 2*parameter_list[0]/math.pi, global_shift= -0.5).on(qubits_a[i], qubits_b[i])

    # Implements the exp(XX) operation on all entangled states
    for i in range(0, qubit_number):
        yield cirq.XXPowGate(exponent= 2*parameter_list[1]/math.pi, global_shift= -0.5).on(qubits_a[i], qubits_b[i])

# Defining the QAOA process

def qaoa_run(qubits_a, qubits_b, depth, qubit_number, gamma_list, alpha_list):

    circuit = cirq.Circuit()
    circuit.append(prepare_entangled_states(qubits_a, qubits_b, qubit_number))

    for j in range(0, depth):

        circuit.append(create_cost_ham(qubits_a, qubit_number, gamma_list))
        #circuit.append(create_cost_ham(qubits_b, qubit_number, gamma_list))
        circuit.append(create_mixer_ham(qubits_a, qubits_b, qubit_number, alpha_list))

    print(circuit)

    #circuit.append(cirq.measure(*qubits, key='x'))
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    return result

#Preparring the TFD state for the cost function

def construct_tfd_state(qubit_number, transverse_field_strength):

    # In this implementation, the eigenvectors of the Hamiltonian and the transposed Hamiltonian are calculated separately

    matrix = calculate_ising_matrix(qubit_number, transverse_field_strength)
    eigen = find_eigenvec_eigenval(matrix)
    partition = calculate_terms_partition(eigen[0])

    print(matrix)
    print(eigen)
    print(partition)

    vec = np.zeros(2**(2*qubit_number))
    for i in range(0, 2**qubit_number):
        addition = (partition[0][i]/partition[1])*(np.kron(eigen[1][i], np.conj(eigen[1][i])))
        vec = np.add(vec, addition)

    return vec

# Defining the cost function

def calculate_cost(list):

    gamma_list = [list[0], list[1]]
    alpha_list = [list[2], list[3]]

    simulated_state = qaoa_run(qubits_a, qubits_b, depth, qubit_number, gamma_list, alpha_list).state_vector()

    good_state = construct_tfd_state(qubit_number, transverse_field_strength)

    cost_int = np.inner(np.conj(good_state), simulated_state)
    cost = cost_int*np.conj(cost_int)

    print(cost)
    print([gamma_list, alpha_list])
    return 1-cost.real

def run_optimization_process():

    init = [random.randint(-600, 600)/100 for i in range(0, 4)]
    out = minimize(calculate_cost, x0=init, method="Nelder-Mead", options={'maxiter':500}, tol=1e-40)
    print(out)
    optimal_param = out['x']

    print("Optimal Parameters: "+str(optimal_param))

    final_final_state = qaoa_run(qubits_a, qubits_b, depth, qubit_number, [optimal_param[i] for i in range(0, 2)], [optimal_param[i] for i in range(2, 4)]).state_vector()

    density_matrix = cirq.density_matrix_from_state_vector(final_final_state)

    print("Optimal Final State: "+str(final_final_state))
    print("Probability Final State: "+str([np.conj(i)*i for i in list(final_final_state)]))

    norm = 0
    for i in list(final_final_state):
        norm = norm + float(i.real)**2
    norm = math.sqrt(norm)

    norm_state = [float(i.real/norm) for i in list(final_final_state)]

    print("Normalized Real: "+str(norm_state))

    good_state = construct_tfd_state(qubit_number, transverse_field_strength)
    print("Target State: "+str(good_state))

    good_density = cirq.density_matrix_from_state_vector(good_state)

    final_cost = np.inner(np.conj(good_state), final_final_state)*np.inner(np.conj(final_final_state), good_state)

    print("Final Cost: "+str(final_cost.real))

    final_cost_absolute = np.inner(good_state, np.array(norm_state))

    #np.inner(np.conj(np.array(norm_state)), good_state)

    print("The Absolute Final Cost: "+str(final_cost_absolute))

    return [density_matrix, good_density]

dm = run_optimization_process()

def create_density_plot(data):

    array = np.array(data)
    plt.matshow(array)
    plt.colorbar()
    plt.title("Density Matrix")
    plt.show()

create_density_plot(dm[0].real)
create_density_plot(dm[1].real)
