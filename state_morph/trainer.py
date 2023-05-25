
from .core import BaseModel
from .utils import _map_step, _reduce_step_wrapper, _random_segment_wrapper, _split_partition, \
    _map_segment, _reduce_segment
from .io import StateMorphIO
import copy
import random
import math
from dask.distributed import as_completed


class StateMorphTrainer(object):
    def __init__(self, client, num_workers, num_state, model_path, model_name,
                 delta=1e-6, patience=10, init_temp=100, final_temp=1e-4, alpha=0.95, schedule='concave', 
                 num_prefix=0, num_suffix=0, affix_lbound=60, stem_ubound=150, bulk_prob = 0.15) -> None:
        assert schedule in ['concave', 'convex', 'linear'], 'Schedule must be one of concave, convex, linear'
        assert 0 <= bulk_prob <= 1, 'Bulk probability must be in [0, 1]'
        assert num_state >= 2, 'Number of state must be greater than 2'
        assert num_state - num_prefix - num_suffix >= 2, 'Number of state must be greater than number of affixes'
        self.client = client
        self.num_state = num_state
        self._delta = delta
        self._patience = patience
        self._final_temp = final_temp
        self._alpha = alpha
        self._current_temp = init_temp
        self.__partitions = None
        self.__model_name = model_name
        self.__schedule = schedule
        self.__num_prefix = num_prefix
        self.__num_suffix = num_suffix
        self.__affix_lbound = affix_lbound
        self.__stem_ubound = stem_ubound
        self.__bulk_prob = bulk_prob
        self.__num_partitions = num_workers
        self.__io = StateMorphIO(model_path + '/' + model_name)
        self.__init_model_param = None
        
    def __checkpoint(self, model, iteration, loss):
        print('Save checkpoint:', iteration, 'Loss:', loss)
        self.__io.write_binary_model_file(model, '{}_{}_{:.4f}.bin'.format(self.__model_name, iteration, loss), no_corpus=True)
        if iteration == 'FINAL':
            self.__io.write_segmented_file(model, '{}_{}_{:.4f}.txt'.format(self.__model_name, iteration, loss))
    
    def load_checkpoint(self, checkpoint_file):
        model = StateMorphIO().load_model_from_binary_file(checkpoint_file)
        self.__init_model_param = model.get_model_dict()
        self.__init_loss = model.compute_encoding_cost()
        
    def load_raw_corpus(self, corpus_file, **kwargs) -> None:
        """Load corpus to state morphology model."""
        num_partitions = sum([_['nthreads'] for _ in self.client.scheduler_info()['workers'].values()])
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.read().splitlines()
            self.__partitions = _split_partition(self.client.scatter(corpus), self.__num_partitions)
            if self.__init_model_param is None:
                partition_with_arg = [
                    (i, partition, self.num_state, self.__num_prefix, self.__num_suffix)
                    for i, partition in enumerate(self.__partitions)
                ]
                futures =  self.client.map(_random_segment_wrapper, partition_with_arg)
                results = [result for _, result in as_completed(futures, with_results=True)]
                reduce_step = self.client.submit(
                    _reduce_step_wrapper(self.num_state, self.__num_prefix, self.__num_suffix), results)
                self.__init_model_param, self.__init_loss = reduce_step.result()
            print('Corpus loaded...')

    def __segment_randomly(self, iteration, total_iteration):
        prob = 0
        if self.__schedule == 'concave':
            prob = math.sqrt(1 - (iteration / total_iteration) ** 2)
        elif self.__schedule == 'convex':
            if iteration < 0.95 * total_iteration:
                prob = 1.0 / (iteration + 1)
            else:
                prob = 0.0
        elif self.__schedule == 'linear':
            prob = 1.0 - iteration / (0.95 * total_iteration)
        if prob > 1.0:
            prob = 1.0
        if prob < 0.0 or not iteration:
            prob = 0.0
        return prob / 2
        
    def __step(self, iteration, model_param, total_iteration):
        print('Iteration:', iteration, '/', total_iteration, 'Temperature:', self._current_temp)
        partition_with_arg = [
            (i, model_param, partition, self._current_temp, self.__segment_randomly(iteration, total_iteration))
            for i, partition in enumerate(self.__partitions)]
        futures =  self.client.map(_map_step, partition_with_arg)
        results = [result for _, result in as_completed(futures, with_results=True)]
        reduce_step = self.client.submit(
                _reduce_step_wrapper(self.num_state, self.__num_prefix, self.__num_suffix), results)
        model_param, loss = reduce_step.result()
        print('Reduce step finished...')
        print('Iteration: {}, Cost: {}'.format(iteration, loss))
        return loss, model_param
    
    def __general_segment(self, model_param):
        partition_with_arg = [(i, model_param, partition) for i, partition in enumerate(self.__partitions)]
        futures =  self.client.map(_map_segment, partition_with_arg)
        results = [result for _, result in as_completed(futures, with_results=True)]
        reduce_step = self.client.submit(_reduce_segment, results)
        segmented_corpus = reduce_step.result()
        return segmented_corpus
    
    def __collect(self, model_param):
        print('Final segmenting started...')
        segmented_corpus = self.__general_segment(model_param)
        print('Final segmenting finished...')
        return segmented_corpus
    
    def __bulk_de_registration(self, model_param):
        print('Bulk de-registration started...')
        deregistered_morph = set()
        __map_key = lambda x: (x[0], int(x[1]))
        for k, count in model_param['lexicon'].items():
            morph, state = __map_key(k.split('_'))
            if (state <= self.__num_prefix or state > self.num_state - self.__num_suffix) and \
                count < self.__affix_lbound and random.random() < self.__bulk_prob:
                deregistered_morph.add((morph, state))
            elif self.__num_prefix < state <= self.num_state - self.__num_suffix and count > self.__stem_ubound and \
                random.random() < self.__bulk_prob:
                deregistered_morph.add((morph, state))
            
        filtered_segmented_corpus = [
            (segment, cost)
            for segment, cost in self.__general_segment(model_param)
            if all(k not in deregistered_morph for k in segment)
        ]
        deregistered_model = BaseModel(model_param)
        deregistered_model.update_segmented_corpus(filtered_segmented_corpus)
        print('Bulk de-registration finished...')
        print('Removed morphs:', len(deregistered_morph))
        return deregistered_model.compute_encoding_cost(), deregistered_model.get_param_dict()
    
    def train(self, iteration=10) -> BaseModel:
        """Train state morphology model."""
        model_param = copy.deepcopy(self.__init_model_param)
        p_loss = -1
        count = 0
        
        temp = math.inf
        if self._final_temp> 0 and self._current_temp > 0 and self._alpha > 0:
            temp = math.ceil((math.log2(self._final_temp) - math.log2(self._current_temp)) / math.log2(self._alpha))        
        total_iteration = min(temp, iteration)
        print('Init cost:', self.__init_loss)
        for _ in range(iteration):
            self._current_temp = max(self._final_temp, self._current_temp * self._alpha)
            loss, model_param = self.__step(_, model_param, total_iteration)
            if random.random() < (math.exp(_/(total_iteration / 3.0)) - 1) / (math.exp(3) - 1):
                loss, model_param = self.__bulk_de_registration(model_param)
            
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
        
        segmented_corpus = self.__collect(model_param)
        new_model = BaseModel(model_param)
        new_model.update_segmented_corpus(segmented_corpus, update_model=False)
        self.__checkpoint(new_model, 'FINAL', new_model.compute_encoding_cost())
        return new_model