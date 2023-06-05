from .core import BaseModel
from .io import StateMorphIO
import random
import socket
import os


def _map_step(args):
    """Map step function for multiprocessing."""
    partition_id, init_model_param, base_path, temperature, random_seg_prob = args
    corpus = StateMorphIO(base_path).load_partition_file(partition_id)
    print('Map ID:', partition_id, 
          'Host:', socket.gethostname(), 
          'PID:', os.getpid(),
          'Corpus size:', len(corpus), 
          'Random segmentation probability:', random_seg_prob,
          'started...')
    model = BaseModel(init_model_param)
    random.shuffle(corpus)
    to_be_trained = corpus[:int(len(corpus) * (1 - random_seg_prob))]
    to_be_random_segmented = corpus[int(len(corpus) * (1 - random_seg_prob)):]
    segmented_corpus = []
    if len(to_be_trained):
        model_param, segmented_corpus = model.train_step(to_be_trained, temperature=temperature)
    if len(to_be_random_segmented):
        random_segmented_corpus = _random_segment(
            to_be_random_segmented, init_model_param['num_state'] - 2, 
            init_model_param['num_prefix'], init_model_param['num_suffix'],
            init_model_param['transition_ctrl'])
        model.update_segmented_corpus(segmented_corpus + random_segmented_corpus)
    model_param = model.get_param_dict()
    print('Map ID:', partition_id, 'ended...')
    return model_param

_reduce_step_wrapper = lambda num_state, num_prefix, num_suffix, transition_ctrl: lambda map_outputs: \
    _reduce_step(map_outputs, num_state, num_prefix, num_suffix, transition_ctrl)
                
def _reduce_step(map_outputs, num_state, num_prefix, num_suffix, transition_ctrl):
    def _reduce(reduced_model_param, model_param):
        """Reduce step function for multiprocessing."""
        total_model_param = reduced_model_param
        
        for k, v in model_param['state_freq'].items():
            if k not in total_model_param['state_freq']:
                total_model_param['state_freq'][k] = 0
            total_model_param['state_freq'][k] = total_model_param['state_freq'][k] + v
        for k, v in model_param['lexicon'].items():
            if k not in total_model_param['lexicon']:
                total_model_param['lexicon'][k] = 0
            total_model_param['lexicon'][k] = total_model_param['lexicon'][k] + v

        if not len(total_model_param['transition_freq']):
            total_model_param['transition_freq'] = model_param['transition_freq']
        else:
            for i in range(len(model_param['transition_freq'])):
                for j in range(len(model_param['transition_freq'][i])):
                    total_model_param['transition_freq'][i][j] = total_model_param['transition_freq'][i][j] + \
                        model_param['transition_freq'][i][j]
        return total_model_param
    
    
    _model_param = {
        'num_state': num_state + 2,
        'num_prefix': num_prefix,
        'num_suffix': num_suffix,
        'lexicon': {},
        'state_freq': {k : 0 for k in range(num_state + 2)},
        'transition_freq': [[0 for _ in range(num_state + 2)] for _ in range(num_state + 2)],
        'transition_ctrl': transition_ctrl
    }

    for model_param in map_outputs:
        _model_param =_reduce(_model_param, model_param)
    _cost = BaseModel(_model_param).compute_encoding_cost()
    return _model_param, _cost        

def _random_segment(corpus, num_state, num_prefix, num_suffix, transition_ctrl):
    segmented_corpus = []
    for word in corpus:
        if len(word) > 1:
            states = []
            while not len(states) or any([not transition_ctrl.get(_, 1) for _ in zip(states[:-1], states[1:])]):
                bounds = sorted(random.sample(list(range(1, len(word))), random.randint(1, len(word) - 1)))
                if bounds[-1] != len(word):
                    bounds.append(len(word))
                if bounds[0] != 0:
                    bounds.insert(0, 0)
                bounds = [(start, end) for start, end in zip(bounds[:-1], bounds[1:]) if start != end]
                prefixes = [random.randint(1, num_prefix) for _ in range(random.randint(0, num_prefix))]
                suffixes = [random.randint(num_state - num_suffix, num_state) for _ in range(random.randint(0, num_suffix))]
                stems = [random.randint(num_prefix + 1, num_state - num_suffix) 
                        for _ in range(len(bounds) - len(prefixes) - len(suffixes))]
                states = prefixes + stems + suffixes
            segment = [
                (word[start:end], state)
                for (start, end), state in zip(bounds, states)
            ]
        else:
            segment = [(word, random.randint(num_prefix + 1, num_state - num_suffix))]
        segmented_corpus.append((segment, 0))
    return segmented_corpus

def _random_segment_wrapper(args) -> list:
    partition_id, base_path, num_state, num_prefix, num_suffix, transition_ctrl = args
    corpus = StateMorphIO(base_path).load_partition_file(partition_id)
    print('Random Seg ID:', partition_id, 
          'Host:', socket.gethostname(), 
          'PID:', os.getpid(),
          'Corpus size:', len(corpus), 
          'started...')
    model_param = {
        'num_state': num_state + 2,
        'num_prefix': num_prefix,
        'num_suffix': num_suffix,
        'lexicon': {},
        'state_freq': {k : 0 for k in range(num_state + 2)},
        'transition_freq': [[0 for _ in range(num_state + 2)] for _ in range(num_state + 2)],
        'transition_ctrl': transition_ctrl
    }
    segmented_corpus = _random_segment(corpus, num_state, num_prefix, num_suffix, transition_ctrl)
    model = BaseModel(model_param)
    model.update_segmented_corpus(segmented_corpus)
    print('Random Seg ID:', partition_id, 'ended...')
    return model.get_param_dict()


def _split_partition(corpus, num_partitions):
    partition_size = len(corpus) // num_partitions
    partitions = [corpus[i:i+partition_size]
                    for i in range(0, len(corpus), partition_size)]
    temp = partitions[:-1]
    if num_partitions > 1:
        temp[-1] += partitions[-1]
    partitions = temp 
    return partitions

def _map_segment(args):
    partition_id, model_param, base_path, is_final = args
    corpus = StateMorphIO(base_path).load_partition_file(partition_id)
    """Map step function for multiprocessing."""
    print('Seg ID:', partition_id, 
          'Host:', socket.gethostname(), 
          'PID:', os.getpid(),
          'Corpus size:', len(corpus), 
          'started...')
    model = BaseModel(model_param)
    _, segmented_corpus = model.train_step(corpus, is_final=is_final)
    costs = [cost for _, cost in segmented_corpus if cost > 0]
    print('Map ID:', partition_id, 'ended...')
    return segmented_corpus, costs

def _reduce_segment(map_outputs):
    """Reduce step function for multiprocessing."""
    _reduce = lambda reduced_corpus, segmented_corpus: reduced_corpus + segmented_corpus
    
    corpus = []
    for segmented_corpus, costs in map_outputs:
        corpus = _reduce(corpus, segmented_corpus)
    return corpus

def _dump_partitions(args):
    partition_id, base_path, partition = args
    """Map step function for multiprocessing."""
    print('Seg ID:', partition_id, 
          'Host:', socket.gethostname(), 
          'PID:', os.getpid(),
          'Corpus size:', len(partition), 
          'started...')
    StateMorphIO(base_path).write_partition_file(partition_id, partition)
    return True