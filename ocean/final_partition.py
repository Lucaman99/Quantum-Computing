from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import dimod
import neal
import math

#Inputted number for partitioning
n = 3

#sampler = EmbeddingComposite(DWaveSampler(endpoint='https://cloud.dwavesys.com/sapi', token='DEV-674dd7ba27484b99e54e0a675f7eadabc4ee797b', solver='DW_2000Q_2_1'))
#Not Gate
query = {('x', 'x'): -1, ('x', 'z'): 2, ('z', 'x'): 0, ('z', 'z'): -1}
#OR Gate
query2 = {('x1', 'x1'): 1, ('x1', 'z'): -2, ('x2', 'x2'): 1, ('x2', 'z'): -2, ('z', 'z'): 1, ('x1', 'x2'):1}
#AND Gate
query4 = {}

#Create loss functions for different number outputs from the circuit

#Create circuit for each possible config of numbers
def circuit_creator(number):

    holder_query = {}

    #["input_1", "input_2", "and", "nand", "or", "final"]

    def add_bits(keywords):
        qubo_matrix = [[1, 2, -2, 0, -2, 0], [0, 1, -2, 0, -2, 0], [0, 0, 2, 2, 0, 0], [0, 0, 0, -1, 0, -2], [0, 0, 0, 1, 1, -2], [0, 0, 0, 0, 0, 3]]
        for i in range(0, 6):
            for k in range(0, 6):
                if ((keywords[i], keywords[k]) not in holder_query.keys()):
                    holder_query.update({(keywords[i], keywords[k]): qubo_matrix[i][k]})
                else:
                    holder_query[(keywords[i], keywords[k])] = holder_query[(keywords[i], keywords[k])] + qubo_matrix[i][k]

    i = 0

    bit_size = "{0:b}".format(number)

    #Create one bit addition
    var_names = []
    recording = []
    carry_bit = 0
    for i in range(0, len(bit_size)):
        key_list = ["input_1"+str(i), "input_2"+str(i), "and"+str(i), "nand"+str(i), "or"+str(i), "final"+str(i)]
        add_bits(key_list)
        var_names.append("final"+str(i))
        var_names.append("and"+str(i))


    o = 0
    while (len(var_names) > 0):
        recording.append(var_names[0])
        var_names.pop(0)
        for h in range (0, math.floor(len(var_names)/2)):
            o = o + 1
            add_bits([var_names[h*2], var_names[(h*2)+1], "and"+str(i+o), "nand"+str(i+o), "or"+str(i+o), "final"+str(i+o)])
            var_names[(h*2)] = "final"+str(i+o)
            var_names[(h*2)+1] = "and"+str(i+o)

    final_min = []


    minimizing = []

    print(recording)


    for a in range(0, len(recording)):
        minimizing.append(-1*(2**a))

    for m in range (0, len(recording)):
        if ((recording[m], recording[m]) in holder_query.keys()):
            holder_query[(recording[m], recording[m])] = holder_query[(recording[m], recording[m])] + minimizing[m]*number*2
        else:
            holder_query.update({(recording[m], recording[m]): minimizing[m]*number*2})



    for re in range (0, len(minimizing)):
        for y in range (0, len(minimizing)):
            if ((recording[re], recording[y]) in holder_query.keys()):
                holder_query[(recording[re], recording[y])] = holder_query[(recording[re], recording[y])] + minimizing[re]*minimizing[y]
            else:
                holder_query.update({(recording[re], recording[y]): minimizing[re]*minimizing[y]})

    #final_min[0] = final_min[0] + final_min[1]
    '''

    for ty in range(0, len(final_min)):
        holder_query[final_min[ty]] = holder_query[final_min[ty]] + final_min[ty]

    '''
    #for ty in range (len(minimizing), len(final_min)):



    return [holder_query, recording, bit_size]

zx = circuit_creator(n)

print(zx[1])




query3 = {('input_1', 'input_2'): 2, ('input_1', 'and_out'): -2, ('input_2', 'and_out'): -2, ('and_out', 'and_out'): 2, ('and_out', 'nand_out'): 2, ('nand_out', 'nand_out'): -1, ('input_1', 'input_1'):1, ('input_2', 'input_2'):1, ('input_2', 'or_out'):-2, ('input_1', 'or_out'):-2, ('or_out', 'or_out'):1, ('final', 'final'):3, ('or_out', 'nand_out'):1, ('or_out', 'final'):-2, ('nand_out', 'final'):-2}
query5 = {('x1', 'x2'): 2, ('x1', 'z'): -2, ('x2', 'z'): -2, ('z', 'a'): 2, ('x2', 'd'):-2, ('x1', 'd'):-2}
J_values = {'x1': 1, 'x2': 1, 'z': 2, 'a': -1, 'd': 1}

anneal_logger = []
anneal_counter = []

solver = neal.SimulatedAnnealingSampler()
#response = sampler.sample_qubo(query3, num_reads=2000, chain_strength=2.0)
response = solver.sample_qubo(zx[0], num_reads=2000)
for datum in response.data(['sample', 'energy', 'num_occurrences']):

    #print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)



    processing = datum.sample, "Energy: ", datum.energy

    trial1 = ""
    for u in range(0, len(zx[1])):
        trial1 = str(processing[0][zx[1][u]]) + trial1
    trial1 = int(trial1, 2)

    trial2 = ""
    trial3 = ""
    for s in range(0, len(zx[2])):
        trial2 = str(processing[0]["input_1"+str(s)]) + trial2
        trial3 = str(processing[0]["input_2"+str(s)]) + trial3
    trial2 = int(trial2, 2)
    trial3 = int(trial3, 2)
    trial = [trial3, trial2, trial1]

    if (trial not in anneal_logger):
        anneal_logger.append(trial)
        anneal_counter.append(1)
    else:
        anneal_counter[anneal_logger.index(trial)] = anneal_counter[anneal_logger.index(trial)] + 1

for a in range(0, len(anneal_logger)):
    print(anneal_logger[a])
    print(anneal_counter[a])
    print("----------------------------------------------")

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
