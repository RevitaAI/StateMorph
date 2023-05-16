
from .core import BaseModel
from .utils import _map_step, _reduce_step_wrapper, _random_segment_wrapper, _split_partition, \
    _map_segment, _reduce_segment
from statistics import mean
import dask
import copy
from dask.distributed import get_client


class StateMorphTrainer(object):
    def __init__(self, num_state, delta=1e-6, patience=10, init_temp=100, final_temp=1e-4, alpha=0.95) -> None:
        self.num_state = num_state
        self._delta = delta
        self._patience = patience
        self._final_temp = final_temp
        self._alpha = alpha
        self._current_temp = init_temp
        self.__partitions = None
        
    
    def load_raw_corpus(self, corpus_file, **kwargs) -> None:
        """Load corpus to state morphology model."""
        client = get_client()
        num_partitions = sum([_['nthreads'] for _ in client.scheduler_info()['workers'].values()])
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.read().splitlines()
            self.__partitions = _split_partition(corpus, num_partitions)
            outputs = []
            for i, partition in enumerate(self.__partitions):
                delayed_map_step = dask.delayed(_random_segment_wrapper)(i, partition, self.num_state)
                outputs.append(delayed_map_step)
            delayed_reduce_step = dask.delayed(_reduce_step_wrapper)(outputs)
            self.__init_model_param, costs = delayed_reduce_step.compute()
            loss = mean(costs) if len(costs) else -1
            print('Init cost:', loss)

    def __step(self, i, model_param):
        print('Iteration:', i, 'Temperature:', self._current_temp)
        map_outputs = []
        for _, partition in enumerate(self.__partitions):
            input_arg = (_, model_param, partition, self.num_state, self._current_temp)
            delayed_map_step = dask.delayed(_map_step)(*input_arg)
            map_outputs.append(delayed_map_step)
        delayed_reduce_step = dask.delayed(_reduce_step_wrapper)(map_outputs)
        model_param, costs = delayed_reduce_step.compute()
        print('Reduce step finished...')
        loss = mean(costs) if len(costs) else -1
        print('Iteration: {}, Cost: {}'.format(i, loss))
        return loss, model_param
    
    def __collect(self, model_param):
        print('Final iteration started...')
        map_outputs = []
        for _, partition in enumerate(self.__partitions):
            input_arg = (_, model_param, partition)
            delayed_map_step = dask.delayed(_map_segment)(*input_arg)
            map_outputs.append(delayed_map_step)
        delayed_reduce_step = dask.delayed(_reduce_segment)(map_outputs)
        segmented_corpus, costs = delayed_reduce_step.compute()
        print('Reduce step finished...')
        loss = mean(costs) if len(costs) else -1
        print('Final iteration done, Cost: {}'.format(loss))
        return loss, segmented_corpus
    
    
    def train(self, iteration=10) -> BaseModel:
        """Train state morphology model."""
        model_param = copy.deepcopy(self.__init_model_param)
        p_loss = -1
        count = 0
        for _ in range(iteration):
            loss, model_param = self.__step(_, model_param)
            
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
        
        loss, segmented_corpus = self.__collect(model_param)
        new_model = BaseModel(model_param)
        new_model.update_segmented_corpus(segmented_corpus, update_model=False)
        return new_model