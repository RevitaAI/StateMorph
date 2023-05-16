from .core import BaseModel
import random
import socket
import os
import copy

def _map_step(partition_id, model_param, segmented_corpus, num_state, temperature):
    """Map step function for multiprocessing."""
    print('Map ID:', partition_id, 
          'Host:', socket.gethostname(), 
          'PID:', os.getpid(),
          'Corpus size:', len(segmented_corpus), 
          'started...')
    model = BaseModel(model_param)
    model.update_segmented_corpus(segmented_corpus, update_model=False)
    model_param, segmented_corpus = model.train_step()
    costs = [cost for _, cost in segmented_corpus if cost > 0]
    random.shuffle(segmented_corpus)
    i = int(len(segmented_corpus) * temperature)
    segmented_corpus = _random_segment(_merge_morph(segmented_corpus[:i]), num_state) + \
                        segmented_corpus[i:]
    print('Map ID:', partition_id, 'ended...')
    return model_param, segmented_corpus, costs

def _reduce_step(reduced_model_param, reduced_corpus, reduced_costs, model_param, segmented_corpus, costs):
    """Reduce step function for multiprocessing."""
    
    total_model_param = copy.deepcopy(reduced_model_param)
    total_corpus = copy.deepcopy(reduced_corpus) + segmented_corpus
    total_costs = copy.deepcopy(reduced_costs) + costs
    
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
    return total_model_param, total_corpus, total_costs
                
def _reduce_step_wrapper(map_outputs):
    total_model_param = {
        'morph_dict': {},
        'state_freq': {},
        'state_size': {},
        'state_char_counts': {},
        'transition_freq': [],
    }
    total_segmented_corpus = []
    total_costs = []
    for model_param, segmented_corpus, costs in map_outputs:
        total_model_param, total_segmented_corpus, total_costs =_reduce_step(
            total_model_param, total_segmented_corpus, total_costs, model_param, segmented_corpus, costs)
    return total_model_param, total_segmented_corpus, total_costs         

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
        'morph_dict': {},
        'state_freq': {},
        'state_size': {},
        'state_char_counts': {},
        'transition_freq': [],
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
    return model.get_param_dict(), segmented_corpus, [0] * len(corpus)

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