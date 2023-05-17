from .core import BaseModel
import random
import socket
import os
import copy

def _map_step(partition_id, model_param, corpus, temperature):
    """Map step function for multiprocessing."""
    print('Map ID:', partition_id, 
          'Host:', socket.gethostname(), 
          'PID:', os.getpid(),
          'Corpus size:', len(corpus), 
          'started...')
    model = BaseModel(model_param)
    model.set_temperature(temperature)
    model_param, segmented_corpus = model.train_step(corpus)
    costs = [cost for _, cost in segmented_corpus if cost > 0]
    print('Map ID:', partition_id, 'ended...')
    return model_param, costs

_reduce_step_wrapper = lambda num_state: lambda map_outputs: _reduce_step(map_outputs, num_state)
                
def _reduce_step(map_outputs, num_state):
    def _reduce(reduced_model_param, reduced_costs, model_param, costs):
        """Reduce step function for multiprocessing."""
        total_model_param = reduced_model_param
        total_costs = reduced_costs + costs
        
        for k, v in model_param['morph_dict'].items():
            if k not in total_model_param['morph_dict']:
                total_model_param['morph_dict'][k] = {}
            for vk, vv in v.items():            
                if vk not in total_model_param['morph_dict'][k]:
                    total_model_param['morph_dict'][k][vk] = 0
                total_model_param['morph_dict'][k][vk] = total_model_param['morph_dict'][k][vk] + vv
        for k, v in model_param['state_freq'].items():
            if k not in total_model_param['state_freq']:
                total_model_param['state_freq'][k] = 0
            total_model_param['state_freq'][k] = total_model_param['state_freq'][k] + v
        for k, v in model_param['state_size'].items():
            if k not in total_model_param['state_size']:
                total_model_param['state_size'][k] = 0
            total_model_param['state_size'][k] = total_model_param['state_size'][k] + v
        for k, v in model_param['state_char_counts'].items():
            if k not in total_model_param['state_char_counts']:
                total_model_param['state_char_counts'][k] = {}
            for vk, vv in v.items():
                if vk not in total_model_param['state_char_counts'][k]:
                    total_model_param['state_char_counts'][k][vk] = 0
                total_model_param['state_char_counts'][k][vk] = total_model_param['state_char_counts'][k][vk] + vv
                
        if not len(total_model_param['transition_freq']):
            total_model_param['transition_freq'] = model_param['transition_freq']
        else:
            for i in range(len(model_param['transition_freq'])):
                for j in range(len(model_param['transition_freq'][i])):
                    total_model_param['transition_freq'][i][j] = total_model_param['transition_freq'][i][j] + \
                        model_param['transition_freq'][i][j]
        return total_model_param, total_costs
    
    
    _model_param = {
        'num_state': num_state + 2,
        'morph_dict': {},
        'state_freq': {k : 0 for k in range(num_state + 2)},
        'state_size': {k : 0 for k in range(num_state + 2)},
        'state_char_counts': {k : {} for k in range(num_state + 2)},
        'transition_freq': [[0 for _ in range(num_state + 2)] for _ in range(num_state + 2)],
    }
    _costs = []
    for model_param, costs in map_outputs:
        _model_param, _costs =_reduce(_model_param, _costs, model_param, costs)
    return _model_param, _costs         

def _random_segment(corpus, num_state):
    segmented_corpus = []
    for word in corpus:
        segment = []
        if len(word) > 1:
            j = 0
            for i in random.sample(list(range(1, len(word))), random.randint(1, len(word) - 1)):
                morph = word[j:i]
                if len(morph) >= 1:
                    segment.append((morph, random.randint(1, num_state)))
                    j = i
            morph = word[j:]
            if len(morph) >= 1:
                segment.append((morph, random.randint(1, num_state)))
        else:
            segment.append((word, random.randint(1, num_state)))
        segmented_corpus.append((segment, 0))
    return segmented_corpus

def _random_segment_wrapper(partition_id, corpus, num_state) -> list:
    print('Random Seg ID:', partition_id, 
          'Host:', socket.gethostname(), 
          'PID:', os.getpid(),
          'Corpus size:', len(corpus), 
          'started...')
    segmented_corpus = []
    model_param = {
        'num_state': num_state + 2,
        'morph_dict': {},
        'state_freq': {k : 0 for k in range(num_state + 2)},
        'state_size': {k : 0 for k in range(num_state + 2)},
        'state_char_counts': {k : {} for k in range(num_state + 2)},
        'transition_freq': [[0 for _ in range(num_state + 2)] for _ in range(num_state + 2)],
    }
    for word in corpus:
        segment = []
        if len(word) > 1:
            j = 0
            for i in random.sample(list(range(1, len(word))), random.randint(1, len(word) - 1)):
                morph = word[j:i]
                if len(morph) >= 1:
                    segment.append((morph, random.randint(1, num_state)))
                    j = i
            morph = word[j:]
            if len(morph) >= 1:
                segment.append((morph, random.randint(1, num_state)))
        else:
            segment.append((word, random.randint(1, num_state)))
        segmented_corpus.append((segment, 0))
    model = BaseModel(model_param)
    model.update_segmented_corpus(segmented_corpus)
    print('Random Seg ID:', partition_id, 'ended...')
    return model.get_param_dict(), [0] * len(corpus)

def _merge_morph(segmented_corpus) -> list:
    corpus = []
    for segment, _ in segmented_corpus:
        word = ''.join([morph for morph, _ in segment])
        corpus.append(word)
    return corpus

def _split_partition(corpus, num_partitions):
    partition_size = len(corpus) // num_partitions
    partitions = [corpus[i:i+partition_size]
                    for i in range(0, len(corpus), partition_size)]
    temp = partitions[:-1]
    temp[-1] += partitions[-1]
    partitions = temp 
    return partitions

def _map_segment(partition_id, model_param, corpus):
    """Map step function for multiprocessing."""
    print('Seg ID:', partition_id, 
          'Host:', socket.gethostname(), 
          'PID:', os.getpid(),
          'Corpus size:', len(corpus), 
          'started...')
    model = BaseModel(model_param)
    _, segmented_corpus = model.train_step(corpus)
    costs = [cost for _, cost in segmented_corpus if cost > 0]
    print('Map ID:', partition_id, 'ended...')
    return segmented_corpus, costs

def _reduce_segment(map_outputs):
    """Reduce step function for multiprocessing."""
    def _reduce(reduced_corpus, reduced_costs, segmented_corpus, costs):
        merged_corpus = reduced_corpus + segmented_corpus
        merged_costs = reduced_costs + costs
        return merged_corpus, merged_costs
    
    
    corpus = []
    total_costs = []
    for segmented_corpus, costs in map_outputs:
        corpus, total_costs = _reduce(corpus, total_costs, segmented_corpus, costs)
    return corpus, total_costs