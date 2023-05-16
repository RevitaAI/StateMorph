
from .core import BaseModel
from .utils import _map_step, _reduce_step_wrapper, _random_segment_wrapper, _split_partition, \
    _map_segment, _reduce_segment
from statistics import mean
import dask
import copy
from dask.distributed import as_completed


class StateMorphTrainer(object):
    def __init__(self, client, num_state, delta=1e-6, patience=10, init_temp=100, final_temp=1e-4, alpha=0.95) -> None:
        self.client = client
        self.num_state = num_state
        self._delta = delta
        self._patience = patience
        self._final_temp = final_temp
        self._alpha = alpha
        self._current_temp = init_temp
        self.__partitions = None
        
    
    def load_raw_corpus(self, corpus_file, **kwargs) -> None:
        """Load corpus to state morphology model."""
        num_partitions = sum([_['nthreads'] for _ in self.client.scheduler_info()['workers'].values()])
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.read().splitlines()
            self.__partitions = _split_partition(corpus, num_partitions)
            futures =  [self.client.submit(_random_segment_wrapper, i, partition, self.num_state) 
                        for i, partition in enumerate(self.__partitions)]
            results = [result for _, result in as_completed(futures, with_results=True)]
            reduce_step = self.client.submit(_reduce_step_wrapper, results)
            self.__init_model_param, costs = reduce_step.result()
            loss = mean(costs) if len(costs) else -1
            print('Init cost:', loss)

    def __step(self, i, model_param):
        print('Iteration:', i, 'Temperature:', self._current_temp)
        futures =  [self.client.submit(_map_step, i, model_param, partition, self.num_state, self._current_temp) 
                        for i, partition in enumerate(self.__partitions)]
        results = [result for _, result in as_completed(futures, with_results=True)]
        reduce_step = self.client.submit(_reduce_step_wrapper, results)
        model_param, costs = reduce_step.result()
        print('Reduce step finished...')
        loss = mean(costs) if len(costs) else -1
        print('Iteration: {}, Cost: {}'.format(i, loss))
        return loss, model_param
    
    def __collect(self, model_param):
        print('Final iteration started...')
        futures =  [self.client.submit(_map_segment, i, model_param, partition) 
                        for i, partition in enumerate(self.__partitions)]
        results = [result for _, result in as_completed(futures, with_results=True)]
        reduce_step = self.client.submit(_reduce_segment, results)
        segmented_corpus, costs = reduce_step.result()
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
            self._current_temp = max(self._final_temp, self._current_temp * self._alpha)
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
        
        loss, segmented_corpus = self.__collect(model_param)
        new_model = BaseModel(model_param)
        new_model.update_segmented_corpus(segmented_corpus, update_model=False)
        return new_model