# This code was from a tutorial I followed on the D-Wave website

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

sampler = EmbeddingComposite(DWaveSampler(endpoint='', token='', solver=''))

query = {('x', 'x'): -1, ('x', 'z'): 2, ('z', 'x'): 0, ('z', 'z'): -1}

response = sampler.sample_qubo(query, num_reads=200)
for datum in response.data(['sample', 'energy', 'num_occurrences']):

    print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
