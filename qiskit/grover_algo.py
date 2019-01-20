from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer

#Grover's Algo to find 2 from 4 possible states

n = 4
q = QuantumRegister(n)
c = ClassicalRegister(n)

circuit = QuantumCircuit(q, c)
#Initialize gates
circuit.x(q[2])
circuit.h(q[0])
circuit.h(q[1])
circuit.h(q[2])
#Apply the oracle
circuit.ccx(q[0], q[1], q[2])
#Apply the Hadamard gates
circuit.h(q[0])
circuit.h(q[1])
#Apply the phase shift
#circuit.cx(q[0], q[2])
#circuit.cx(q[1], q[2])
circuit.x(q[0])
circuit.x(q[1])
circuit.ccx(q[0], q[1], q[2])
circuit.x(q[0])
circuit.x(q[1])
#Apply the second round of Hadamard gates
circuit.h(q[0])
circuit.h(q[1])
circuit.h(q[2])

circuit.measure(q, c)

print(circuit)

backend = Aer.get_backend('qasm_simulator')
job_sim = execute(circuit, backend)
sim_result = job_sim.result()

print(sim_result.get_counts(circuit))
