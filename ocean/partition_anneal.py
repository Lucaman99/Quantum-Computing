from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import dimod
import neal

#sampler = EmbeddingComposite(DWaveSampler(endpoint='https://cloud.dwavesys.com/sapi', token='DEV-674dd7ba27484b99e54e0a675f7eadabc4ee797b', solver='DW_2000Q_2_1'))
#Not Gate
query = {('x', 'x'): -1, ('x', 'z'): 2, ('z', 'x'): 0, ('z', 'z'): -1}
#OR Gate
query2 = {('x1', 'x1'): 1, ('x1', 'z'): -2, ('x2', 'x2'): 1, ('x2', 'z'): -2, ('z', 'z'): 1, ('x1', 'x2'):1}
#AND Gate
query4 = {}

query3 = {('x1', 'x2'): 2, ('x1', 'z'): -2, ('x2', 'z'): -2, ('z', 'z'): 2, ('z', 'a'): 2, ('a', 'a'): -1, ('x1', 'x1'):1, ('x2', 'x2'):1, ('x2', 'd'):-2, ('x1', 'd'):-2, ('d', 'd'):1, ('b', 'b'):2, ('d', 'a'):1, ('d', 'b'):-2, ('a', 'b'):-2, ('b', 'z'):4}
query5 = {('x1', 'x2'): 2, ('x1', 'z'): -2, ('x2', 'z'): -2, ('z', 'a'): 2, ('x2', 'd'):-2, ('x1', 'd'):-2}
J_values = {'x1': 1, 'x2': 1, 'z': 2, 'a': -1, 'd': 1}

anneal_logger = []
anneal_counter = []

solver = neal.SimulatedAnnealingSampler()
response = solver.sample_qubo(query3, num_reads=2000)
for datum in response.data(['sample', 'energy', 'num_occurrences']):

    processing = datum.sample, "Energy: ", datum.energy
    trial = str(processing)
    if (trial not in anneal_logger):
        anneal_logger.append(trial)
        anneal_counter.append(1)
    else:
        anneal_counter[anneal_logger.index(trial)] = anneal_counter[anneal_logger.index(trial)] + 1

for a in range(0, len(anneal_logger)):
    print(anneal_logger[a])
    print(anneal_counter[a])
    print("--------------")

print(list(query3.keys())[0])
print(list(query3.values())[0])


'''
response = sampler.sample_qubo(query2, num_reads=2000)
for datum in response.data(['sample', 'energy', 'num_occurrences']):

    print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
'''
'''
{'z': 1, 'a': 0, 'x1': 1, 'x2': 1, 'd': 1} Energy:  -1.0 Occurrences:  1619
{'z': 1, 'a': 0, 'x1': 1, 'x2': 1, 'd': 1} Energy:  -1.0 Occurrences:  22
{'z': 0, 'a': 1, 'x1': 1, 'x2': 0, 'd': 1} Energy:  -1.0 Occurrences:  74
{'z': 0, 'a': 1, 'x1': 0, 'x2': 0, 'd': 0} Energy:  -1.0 Occurrences:  135
{'z': 0, 'a': 1, 'x1': 0, 'x2': 1, 'd': 1} Energy:  -1.0 Occurrences:  145
'''
