
from .core import BaseModel
from .utils import _map_step, _reduce_step_wrapper, _random_segment_wrapper, _split_partition, \
    _map_segment, _reduce_segment
from .io import StateMorphIO
from statistics import mean
import dask
import copy
from dask.distributed import as_completed


class StateMorphTrainer(object):
    def __init__(self, client, num_state, model_path, model_name,
                 delta=1e-6, patience=10, init_temp=100, final_temp=1e-4, alpha=0.95) -> None:
        self.client = client
        self.num_state = num_state
        self._delta = delta
        self._patience = patience
        self._final_temp = final_temp
        self._alpha = alpha
        self._current_temp = init_temp
        self.__partitions = None
        self.__model_name = model_name
        self.__io = StateMorphIO(model_path + '/' + model_name)
        
    def __checkpoint(self, model, iteration, loss):
        print('Save checkpoint:', iteration, 'Loss:', loss)
        self.__io.write_binary_model_file(model, '{}_{}_{:.4f}.bin'.format(self.__model_name, iteration, loss), no_corpus=True)
        if iteration == 'FINAL':
            self.__io.write_segmented_file(model, '{}_{}_{:.4f}.txt'.format(self.__model_name, iteration, loss))
        
    def load_raw_corpus(self, corpus_file, **kwargs) -> None:
        """Load corpus to state morphology model."""
        num_partitions = sum([_['nthreads'] for _ in self.client.scheduler_info()['workers'].values()])
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.read().splitlines()
            self.__partitions = self.client.scatter(_split_partition(corpus, num_partitions))
            futures =  [self.client.submit(_random_segment_wrapper, i, partition, self.num_state) 
                        for i, partition in enumerate(self.__partitions)]
            results = [result for _, result in as_completed(futures, with_results=True)]
            reduce_step = self.client.submit(_reduce_step_wrapper(self.num_state), results)
            self.__init_model_param, loss = reduce_step.result()
            print('Init cost:', loss)

    def __step(self, i, model_param):
        print('Iteration:', i, 'Temperature:', self._current_temp)
        scattered_model_param = [model_param] * len(self.__partitions) # self.client.scatter([model_param] * len(self.__partitions))
        futures =  [self.client.submit(_map_step, i, mp, partition, self._current_temp) 
                    for i, (partition, mp) in enumerate(zip(self.__partitions, scattered_model_param))]
        results = [result for _, result in as_completed(futures, with_results=True)]
        reduce_step = self.client.submit(_reduce_step_wrapper(self.num_state), results)
        model_param, loss = reduce_step.result()
        print('Reduce step finished...')
        print('Iteration: {}, Cost: {}'.format(i, loss))
        return loss, model_param
    
    def __collect(self, model_param):
        print('Final segmenting started...')
        scattered_model_param = self.client.scatter([model_param] * len(self.__partitions))
        futures =  [self.client.submit(_map_segment, i, mp, partition) 
                    for i, (partition, mp) in enumerate(zip(self.__partitions, scattered_model_param))]
        results = [result for _, result in as_completed(futures, with_results=True)]
        reduce_step = self.client.submit(_reduce_segment, results)
        segmented_corpus = reduce_step.result()
        print('Final segmenting finished...')
        return segmented_corpus
    
    
    def train(self, iteration=10) -> BaseModel:
        """Train state morphology model."""
        model_param = copy.deepcopy(self.__init_model_param)
        p_loss = -1
        count = 0
        try:
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
                    if p_loss > loss:
                        self.__checkpoint(BaseModel(model_param), _, loss)
                    count = 0
                    p_loss = loss
        except Exception:
            print('Fallback...')
        
        segmented_corpus = self.__collect(model_param)
        new_model = BaseModel(model_param)
        new_model.update_segmented_corpus(segmented_corpus, update_model=False)
        self.__checkpoint(new_model, 'FINAL', new_model.compute_encoding_cost())
        return new_model