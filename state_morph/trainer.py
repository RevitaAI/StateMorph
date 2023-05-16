
from .core import BaseModel
from .utils import _map_step, _reduce_step_wrapper, _random_segment_wrapper, _concat_list, _split_partition
from statistics import mean
import dask
from dask.distributed import get_client


class StateMorphTrainer(object):
    def __init__(self, num_state, delta=1e-6, patience=10, init_temp=100, final_temp=1e-4, alpha=0.95) -> None:
        self.num_state = num_state
        self._delta = delta
        self._patience = patience
        self._final_temp = final_temp
        self._alpha = alpha
        self._current_temp = init_temp
        
    
    def load_raw_corpus(self, corpus_file, **kwargs) -> None:
        """Load corpus to state morphology model."""
        client = get_client()
        num_partitions = sum([_['nthreads'] for _ in client.scheduler_info()['workers'].values()])
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.read().splitlines()
            partitions = _split_partition(corpus, num_partitions)
            outputs = []
            for i, partition in enumerate(partitions):
                segmented_partition = dask.delayed(_random_segment_wrapper)(i, partition, self.num_state)
                outputs.append(segmented_partition)
            segmented_corpus = dask.delayed(_concat_list)(outputs).compute()
            
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
    
    def __step(self, i, num_partitions, model_param, segmented_corpus):
        print('Iteration:', i, 'Temperature:', self._current_temp)
        partitions = _split_partition(segmented_corpus, num_partitions)
        partitions = [(_, model_param, partition, self.num_state, self._current_temp) 
                        for _, partition in enumerate(partitions)]
        
        map_outputs = []
        for partition in partitions:
            delayed_map_step = dask.delayed(_map_step)(*partition)
            map_outputs.append(delayed_map_step)
        delayed_reduce_step = dask.delayed(_reduce_step_wrapper)(map_outputs)
        model_param, segmented_corpus, costs = delayed_reduce_step.compute()
        print('Reduce step finished...')
        loss = mean(costs) if len(costs) else -1
        print('Iteration: {}, Cost: {}'.format(i, loss))
        return loss, model_param, segmented_corpus
    
    def train(self, iteration=10) -> BaseModel:
        """Train state morphology model."""
        model_param = self._base_model.get_param_dict()
        segmented_corpus = self._base_model.segmented_corpus
        p_loss = -1
        count = 0
        client = get_client()
        num_partitions = sum([_['nthreads'] for _ in client.scheduler_info()['workers'].values()])
        for _ in range(iteration):
            loss, model_param, segmented_corpus = self.__step(_, num_partitions, model_param, segmented_corpus)
            
            # Early stopping
            if abs(p_loss - loss) < self._delta and loss:
                count += 1
                if count == self._patience:
                    print('Early stopping...')
                    break
            elif self._current_temp < self._final_temp:
                break
            else:
                count = 0
                p_loss = loss
                
            
            self._current_temp = max(self._final_temp, self._current_temp * self._alpha)
            # client.restart()
        self._current_temp = 0
        loss, model_param, segmented_corpus = self.__step('FINAL', num_partitions, model_param, segmented_corpus)
        new_model = BaseModel(model_param)
        new_model.update_segmented_corpus(segmented_corpus, update_model=False)
        return new_model