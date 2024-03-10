from .core import BaseModel
from .io import StateMorphIO
import random
import socket
import os
import logging

empty_model_param = lambda num_state, num_prefix, num_suffix, transition_ctrl: {
    'num_state': num_state,
    'num_prefix': num_prefix,
    'num_suffix': num_suffix,
    'lexicon': {},
    'state_freq': {k : 0 for k in range(num_state)},
    'transition_freq': [[0 for _ in range(num_state)] for _ in range(num_state)],
    'transition_ctrl': transition_ctrl
}

log_wrapper = lambda logger, log: logging.getLogger(logger).info(log)

def _map_step(args):
    """Map step function for multiprocessing."""
    partition_id, base_path, temperature, random_seg_prob = args
    io = StateMorphIO(base_path)
    corpus = io.load_partition_file(partition_id)
    init_model_param = io.load_temp_model_params()
    log = 'Map ID: {} Host: {} PID: {} Corpus size: {} Random segmentation probability: {} started...'.format(
        partition_id, socket.gethostname(), os.getpid(), len(corpus), random_seg_prob
    )
    log_wrapper("distributed.worker", log)
    model = BaseModel(init_model_param)
    random.shuffle(corpus)
    to_be_trained = corpus[:int(len(corpus) * (1 - random_seg_prob))]
    to_be_random_segmented = corpus[int(len(corpus) * (1 - random_seg_prob)):]
    segmented_corpus = []
    if len(to_be_trained):
        model_param, segmented_corpus = model.train_step(to_be_trained, temperature=temperature)
    if len(to_be_random_segmented):
        random_segmented_corpus = _random_segment(
            to_be_random_segmented, init_model_param['num_state'], 
            init_model_param['num_prefix'], init_model_param['num_suffix'],
            init_model_param['transition_ctrl'])
        model.update_segmented_corpus(segmented_corpus + random_segmented_corpus)
    model_param = model.get_param_dict()
    log_wrapper("distributed.worker", 'Map ID: {} ended...'.format(partition_id))
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
    
    
    _model_param = empty_model_param(num_state, num_prefix, num_suffix, transition_ctrl)

    for model_param in map_outputs:
        _model_param =_reduce(_model_param, model_param)
    _cost = BaseModel(_model_param).compute_encoding_cost()
    return _model_param, _cost        

def _random_segment(corpus, num_state, num_prefix, num_suffix, transition_ctrl):
    segmented_corpus = []
    for word in corpus:
        if len(word) > 1:
            states = []
            while not len(states) or any([not transition_ctrl.get(_, 1) 
                                          for _ in zip([0] + states, states + [num_state - 1])]):
                bounds = sorted(random.sample(list(range(1, len(word))), random.randint(1, len(word) - 1)))
                if bounds[-1] != len(word):
                    bounds.append(len(word))
                if bounds[0] != 0:
                    bounds.insert(0, 0)
                bounds = [(start, end) for start, end in zip(bounds[:-1], bounds[1:]) if start != end]
                prefixes = [random.randint(1, num_prefix) for _ in range(random.randint(0, num_prefix))]
                suffixes = [random.randint(num_state - num_suffix - 2 , num_state - 2 )
                            for _ in range(random.randint(0, num_suffix))]
                stems = [random.randint(num_prefix + 1, num_state - num_suffix - 2) 
                        for _ in range(len(bounds) - len(prefixes) - len(suffixes))]
                states = prefixes + stems + suffixes
            segment = [
                (word[start:end], state)
                for (start, end), state in zip(bounds, states)
            ]
        else:
            segment = [(word, random.randint(num_prefix + 1, num_state - num_suffix - 2))]
        segmented_corpus.append((segment, 0))
    return segmented_corpus

def _random_segment_wrapper(args) -> list:
    partition_id, base_path, num_state, num_prefix, num_suffix, transition_ctrl = args
    corpus = StateMorphIO(base_path).load_partition_file(partition_id)
    log = 'Random Seg ID: {} Host: {} PID: {} Corpus size: {} started...'.format(
        partition_id, socket.gethostname(), os.getpid(), len(corpus)
    )
    log_wrapper("distributed.worker", log)
    model_param = empty_model_param(num_state, num_prefix, num_suffix, transition_ctrl)
    segmented_corpus = _random_segment(corpus, num_state, num_prefix, num_suffix, transition_ctrl)
    model = BaseModel(model_param)
    model.update_segmented_corpus(segmented_corpus)
    log_wrapper("distributed.worker", 'Random Seg ID: {} ended...'.format(partition_id))
    return model.get_param_dict()


def _split_partition(corpus, num_partitions):
    left_over = len(corpus) % num_partitions
    partition_size = (len(corpus) - left_over) // num_partitions
    partitions = [corpus[i:i+partition_size]
                    for i in range(0, (len(corpus) - left_over), partition_size)]
    if left_over:
        for i, obj in enumerate(corpus[-left_over:]):
            partitions[i].append(obj)
    return partitions

def _map_segment(args):
    partition_id, base_path = args
    io = StateMorphIO(base_path)
    corpus = io.load_partition_file(partition_id)
    model_param = io.load_temp_model_params()
    remaining_morphs = io.load_temp_file('remaining_morphs')
    """Map step function for multiprocessing."""
    log = 'Seg ID: {} Host: {} PID: {} Corpus size: {} started...'.format(
        partition_id, socket.gethostname(), os.getpid(), len(corpus)
    )
    log_wrapper("distributed.worker", log)
    model = BaseModel(model_param)
    _, segmented_corpus = model.train_step(corpus, is_final=True)
    costs = [cost for _, cost in segmented_corpus if cost > 0]
    pruned_segmented_corpus = []
    if len(remaining_morphs):
        for (segments, cost) in segmented_corpus:
            if set(segments).issubset(remaining_morphs):
                pruned_segmented_corpus.append((segments, cost))
            else:
                try:
                    pruned_segmented_corpus.append(model.segment(''.join([morph for morph, _ in segments])))
                except AssertionError:
                    pass
    log_wrapper("distributed.worker", 'Map ID: {} ended...'.format(partition_id))
    return segmented_corpus, pruned_segmented_corpus, costs

def _reduce_segment(map_outputs):
    """Reduce step function for multiprocessing."""
    _reduce = lambda reduced_corpus, segmented_corpus: reduced_corpus + segmented_corpus
    
    corpus = []
    pruned_corpus = []
    for segmented_corpus, pruned_segment_corpus, costs in map_outputs:
        corpus = _reduce(corpus, segmented_corpus)
        pruned_corpus = _reduce(pruned_corpus, pruned_segment_corpus)
    return corpus, pruned_corpus

def _dump_partitions(args):
    partition_id, base_path, partition = args
    """Map step function for multiprocessing."""
    log = 'Dump ID: {} Host: {} PID: {} Corpus size: {} started...'.format(
        partition_id, socket.gethostname(), os.getpid(), len(partition)
    )
    log_wrapper("distributed.worker", log)
    StateMorphIO(base_path).write_partition_file(partition_id, partition)
    log_wrapper("distributed.worker", 'Dump ID: {} ended...'.format(partition_id))
    return True