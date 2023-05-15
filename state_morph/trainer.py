
from .core import BaseModel
from .utils import _map_step, _reduce_step_wrapper, _random_segment, _merge_morph
from statistics import mean
import random
import dask



class StateMorphTrainer(object):
    def __init__(self, num_state, delta=1e-6, patience=10, init_temp=100, final_temp=1e-4, alpha=0.95, cluster=None) -> None:
        self.num_state = num_state
        self._delta = delta
        self._patience = patience
        self._final_temp = final_temp
        self._alpha = alpha
        self._current_temp = init_temp
        
    
    def load_raw_corpus(self, corpus_file, **kwargs) -> None:
        """Load corpus to state morphology model."""
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.read().splitlines()
            segmented_corpus = _random_segment(corpus, self.num_state)
            model_params = {
                'morph_dict':  {},
                'state_freq': {},
                'state_size': {},
                'state_char_counts': {},
                'transition_freq': [],
            }
            model = BaseModel(model_params, **kwargs)
            model.update_segmented_corpus(segmented_corpus)
            self._base_model = model
    
    def train(self, iteration=10, num_processes=5) -> BaseModel:
        """Train state morphology model."""
        model_param = self._base_model.get_param_dict()
        segmented_corpus = self._base_model.segmented_corpus
        p_loss = -1
        count = 0
        for _ in range(iteration):
            print('Iteration:', _, 'Temperature:', self._current_temp)
            partition_size = len(segmented_corpus) // num_processes
            partitions = [segmented_corpus[i:i+partition_size]
                         for i in range(0, len(segmented_corpus), partition_size)]
            temp = partitions[:-1]
            temp[-1] += partitions[-1]
            partitions = temp 
            partitions = [(_, model_param, partition) for _, partition in enumerate(partitions)]
            
            with dask.config.set(num_workers=num_processes):
                map_outputs = []
                for partition in partitions:
                    delayed_map_step = dask.delayed(_map_step)(*partition)
                    map_outputs.append(delayed_map_step)
                delayed_reduce_step = dask.delayed(_reduce_step_wrapper)(map_outputs)
                model_param, segmented_corpus = delayed_reduce_step.compute()
            print('Reduce step finished...')
            loss = mean([cost for _, cost in segmented_corpus if cost > 0])
            print('Iteration: {}, Cost: {}'.format(_, loss))
            
            # Early stopping
            if abs(p_loss - loss) < self._delta:
                count += 1
                if count == self._patience:
                    print('Early stopping...')
                    break
            else:
                count = 0
                p_loss = loss
                
            random.shuffle(segmented_corpus)
            i = int(len(segmented_corpus) * self._current_temp)
            segmented_corpus = _random_segment(_merge_morph(segmented_corpus[:i]), self.num_state) + \
                segmented_corpus[i:]
            self._current_temp = max(self._final_temp, self._current_temp * self._alpha)
            
        new_model = BaseModel(model_param)
        new_model.update_segmented_corpus(segmented_corpus, update_model=False)
        return new_model