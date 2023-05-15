
from .core import BaseModel
import random
from multiprocessing import Pool

def _map_step(partition_id, model_param, segmented_corpus):
    """Map step function for multiprocessing."""
    print('Partition', partition_id, 'started...')
    model = BaseModel(model_param)
    model.update_segmented_corpus(segmented_corpus, update_model=False)
    model_param, segmented_corpus = model.train_step()
    return model_param, segmented_corpus

def _reduce_step(total_model_param, total_corpus, model_param, segmented_corpus):
    """Reduce step function for multiprocessing."""
    
    total_corpus += segmented_corpus
    
    for k, v in model_param['morph_dict'].items():
        if k not in total_model_param['morph_dict']:
            total_model_param['morph_dict'][k] = {}
        for vk, vv in v.items():            
            if vk not in total_model_param['morph_dict'][k]:
                total_model_param['morph_dict'][k][vk] = 0
            total_model_param['morph_dict'][k][vk] += vv
    for k, v in model_param['state_freq'].items():
        if k not in total_model_param['state_freq']:
            total_model_param['state_freq'][k] = 0
        total_model_param['state_freq'][k] += v
    for k, v in model_param['state_size'].items():
        if k not in total_model_param['state_size']:
            total_model_param['state_size'][k] = 0
        total_model_param['state_size'][k] += v
    for k, v in model_param['state_char_counts'].items():
        if k not in total_model_param['state_char_counts']:
            total_model_param['state_char_counts'][k] = {}
        for vk, vv in v.items():
            if vk not in total_model_param['state_char_counts'][k]:
                total_model_param['state_char_counts'][k][vk] = 0
            total_model_param['state_char_counts'][k][vk] += vv
            
    if not len(total_model_param['transition_freq']):
        total_model_param['transition_freq'] = model_param['transition_freq']
    else:
        for i in range(len(model_param['transition_freq'])):
            for j in range(len(model_param['transition_freq'][i])):
                total_model_param['transition_freq'][i][j] += model_param['transition_freq'][i][j]


class StateMorphTrainer(object):
    def __init__(self, model) -> None:
        self.base_model = model
    
    def train(self, iteration=10, num_processes=5) -> BaseModel:
        """Train state morphology model."""
        model_param = self.base_model.get_param_dict()
        segmented_corpus = self.base_model.segmented_corpus
        with Pool(num_processes) as p:
            for _ in range(iteration):
                
                random.shuffle(segmented_corpus)
                partition_size = len(segmented_corpus) // num_processes
                partition = [(i, model_param, segmented_corpus[i:i+partition_size])
                             for i in range(0, len(segmented_corpus), partition_size)]
                
                total_model_param = {
                    'morph_dict': {},
                    'state_freq': {},
                    'state_size': {},
                    'state_char_counts': {},
                    'transition_freq': [],
                }
                total_segmented_corpus = []
                for model_param, segmented_corpus in p.map(_map_step, partition):
                    _reduce_step(total_model_param, total_segmented_corpus, model_param, segmented_corpus)
                
                model_param = total_model_param
                segmented_corpus = total_segmented_corpus
                
                
                
                print('Iteration: {}, Cost: {}'.format(_, sum([cost for _, cost in segmented_corpus])))
        new_model = BaseModel(model_param)
        new_model.update_segmented_corpus(segmented_corpus, update_model=False)
        return new_model