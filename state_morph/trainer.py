
from .core import BaseModel
from .utils import _map_step, _reduce_step_wrapper, _random_segment_wrapper, _split_partition, \
    _map_segment, _reduce_segment, _dump_partitions, log_wrapper
from .io import StateMorphIO
from dask.distributed import Client
import random
import math
from copy import deepcopy
from collections import Counter

class StateMorphTrainer(object):
    def __init__(self, client: Client, num_workers: int, num_state: int, model_path: str, model_name: str,
                 delta=1e-6, patience=10, init_temp=100, final_temp=1e-4, alpha=0.95, schedule='concave', 
                 num_prefix=0, num_suffix=0, affix_lbound=60, stem_ubound=150, bulk_prob = 0.15,
                 transition_ctrl={}, charset=None, min_lexicon_freq=0, min_pruned_remain=0, 
                 has_word_boundary=False) -> None:
        '''
        The trainer class for StateMorph model. 
        This class is responsible for training the model.
        It should be instantiated with a client object, which is used to training with Dask.
        
        Parameters
        ----------
        client: dask.distributed.Client
            The client object used for training with Dask
        num_workers: int
            Number of workers to use for training
        num_state: int
            Number of states in the model, including the start and end states
        model_path: str
            Path to the model directory
        model_name: str
            Name of the model
        delta: float
            The minimum change in loss to be considered as improvement.
            If the change in loss is less than delta, the training will stop.
        patience: int
            The number of epochs to wait for improvement before early stopping.
        init_temp: float
            The initial temperature for simulated annealing. Default is 100.
        final_temp: float
            The final temperature for simulated annealing. Default is 1e-4.
        alpha: float
            The decay rate for simulated annealing. Default is 0.95.
        schedule: str
            The schedule for word-level simulated annealing (random segmentation). 
            It can be one of 'concave', 'convex', 'linear'. Default is 'concave'.
        num_prefix: int
            Number of prefix states. Default is 0.
        num_suffix: int
            Number of suffix states. Default is 0.
        affix_lbound: int
            The lower bound of the length of affixes. Default is 60.
        stem_ubound: int
            The upper bound of the length of stems. Default is 150.
        bulk_prob: float
            The probability of bulk de-registration. Default is 0.15.
        transition_ctrl: dict
            The transition control dictionary. Default is empty.
            The key is a tuple of (from_state, to_state), 
            and the value 1 means the transition is allowed, 0 means not allowed.
        charset: set
            The character set dictionary. Default is empty.
            Predefine character set so that model can have correct loss during training if character set is too big.
        min_lexicon_freq: int
            The minimum frequency of lexicon to be considered. Default is 0.
            The lexicon with frequency less than min_lexicon_freq will be pruned.
        min_pruned_remain: int
            The minimum number of pruned morphs to remain. Default is 0.
            Stop pruning if the number of remaining morphs is less than min_pruned_remain.
            If 0, the pruning will not stop until the frequency of lexicon is less than min_lexicon_freq.
        has_word_boundary: bool
            Assign word boundary in segmentation. Default is False.
            If True, the segmentation will include word boundary "▁" in the first morph of each word.
            
            
        Examples
        --------
        >>> from dask.distributed import Client, LocalCluster
        >>> from state_morph import StateMorphTrainer
        >>> client = Client(LocalCluster(n_workers=4))
        >>> trainer = StateMorphTrainer(client, 4, 10, 'model', 'model')
        >>> trainer.load_raw_corpus('corpus.txt')
        >>> trainer.train()
        
        
        '''
        
        
        assert schedule in ['concave', 'convex', 'linear'], 'Schedule must be one of concave, convex, linear'
        assert 0 <= bulk_prob <= 1, 'Bulk probability must be in [0, 1]'
        assert num_state >= 3, 'Number of state must be greater than 3'
        assert num_state - num_prefix - num_suffix >= 3, 'Number of state must be greater than number of affixes'
        assert 0 <= min_lexicon_freq, 'Minimum lexicon frequency must be greater than 0'
        self.client = client
        self.num_state = num_state
        self._delta = delta
        self._patience = patience
        self._final_temp = final_temp
        self._alpha = alpha
        self._current_temp = init_temp
        self.__model_name = model_name
        self.__schedule = schedule
        self.__num_prefix = num_prefix
        self.__num_suffix = num_suffix
        self.__affix_lbound = affix_lbound
        self.__stem_ubound = stem_ubound
        self.__bulk_prob = bulk_prob
        self.__transition_ctrl = transition_ctrl
        self.__num_partitions = num_workers
        self.__min_lexicon_freq = min_lexicon_freq
        self.__min_pruned_remain = min_pruned_remain
        self.__charset_size = charset is not None and len(charset) or 0
        self.__has_word_boundary = has_word_boundary
        self.__io = StateMorphIO(model_path + '/' + model_name, charset=charset)
        self.__init_model_param = None
        
    def __checkpoint(self, model, iteration, loss, save_corpus=False):
        log_wrapper("distributed.scheduler", 'Save checkpoint: {} Loss: {:.4f}'.format(iteration, loss))
        self.__io.write_binary_model_file(
            model, '{}_{}_{:.4f}.bin'.format(self.__model_name, iteration, loss), no_corpus=not save_corpus)
        if iteration == 'FINAL':
            self.__io.write_segmented_file(model, '{}_{}_{:.4f}.txt'.format(self.__model_name, iteration, loss))
    
    def load_checkpoint(self, checkpoint_file) -> None:
        '''
        Load a checkpoint file to resume training.
        
        Parameters
        ----------
        checkpoint_file: str
            Path to the checkpoint file.
        '''
        
        model = StateMorphIO().load_model_from_binary_file(checkpoint_file)
        self.__init_model_param = model.get_param_dict()
        self.__init_loss = model.compute_encoding_cost()
        
    def load_raw_corpus(self, corpus_file, **kwargs) -> None:
        """
        Load corpus to StateMorph model.
        
        Parameters
        ----------
        corpus_file: str
            Path to the corpus file.
        
        """
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.read().lower().splitlines()
            random.shuffle(corpus)
            if self.__charset_size == 0:
                charset = set(''.join(corpus))
                self.__io.set_charset(charset)
                self.__charset_size = len(charset)
            else:
                charset = self.__io.get_charset()
                corpus = list(filter(lambda x: set(x).issubset(charset), corpus))
            log_wrapper("distributed.scheduler", 'Charset size: {}'.format(self.__charset_size))
            __partitions = _split_partition(corpus, self.__num_partitions)
            partitions = self.client.scatter([ (i, self.__io.base_path, partition)
                for i, partition in enumerate(__partitions)
            ])
            self.__io.create_temporary_directory()
            futures = self.client.map(_dump_partitions, partitions)
            results = self.client.gather(futures)
            assert sum(results) == self.__num_partitions, 'Dumping partitions failed'
            if self.__init_model_param is None:
                partition_func = lambda i: (
                    i, 
                    self.__io.base_path, 
                    self.num_state, 
                    self.__num_prefix, 
                    self.__num_suffix, 
                    self.__transition_ctrl,
                    self.__has_word_boundary
                )
                partition_with_arg = self.client.scatter([partition_func(i) for i in range(self.__num_partitions)])
                futures =  self.client.map(_random_segment_wrapper, partition_with_arg)
                reduce_step = self.client.submit(
                    _reduce_step_wrapper(
                        self.num_state, self.__num_prefix, self.__num_suffix, 
                        self.__transition_ctrl, self.__has_word_boundary), 
                    futures)
                self.__init_model_param, self.__init_loss = reduce_step.result()
            self.__io.write_temp_model_params(self.__init_model_param)
            log_wrapper("distributed.scheduler", 'Corpus loaded: {}'.format(len(corpus)))
    
    def __segment_randomly(self, iteration, total_iteration):
        prob = 0
        if iteration < 0.95 * total_iteration:
            if self.__schedule == 'concave':
                prob = math.sqrt(1 - (iteration /  (0.95 * total_iteration)) ** 2)
            elif self.__schedule == 'convex':
                prob = 1.0 / ( 0.95 * iteration + 1)
            elif self.__schedule == 'linear':
                prob = 1.0 - iteration / (0.95 * total_iteration)
        if prob > 1.0:
            prob = 1.0
        if prob < 0.0 or not iteration:
            prob = 0.0
        return prob / 2
        
    def __step(self, iteration, total_iteration):
        log_wrapper("distributed.scheduler", 
                    'Iteration: {} / {} Temperature: {}'.format(iteration, total_iteration, self._current_temp))
        random_seg_prob = self.__segment_randomly(iteration, total_iteration)
        partition_with_arg = self.client.scatter([
            (i, self.__io.base_path, self._current_temp, random_seg_prob)
            for i in range(self.__num_partitions)])
        futures =  self.client.map(_map_step, partition_with_arg)
        reduce_step = self.client.submit(
                    _reduce_step_wrapper(
                        self.num_state, self.__num_prefix, self.__num_suffix, 
                        self.__transition_ctrl, self.__has_word_boundary), 
                    futures)
        model_param, loss = reduce_step.result()
        self.__io.write_temp_model_params(model_param)
        log_wrapper("distributed.scheduler", 'Reduce step finished...')
        log_wrapper("distributed.scheduler", 'Iteration: {}, Cost: {}'.format(iteration, loss))
        return loss, model_param
        
    def __collect(self, model_param):
        __map_key = lambda x: (x[0], int(x[1]))
        log_wrapper("distributed.scheduler", 'Final segmenting started...')
        pruned_model_param = None
        pruned_segmented_corpus = []
        if self.__min_lexicon_freq:
            freq_dict = Counter(model_param['lexicon'].values())
            total = len(model_param['lexicon'])
                
            pruned_size = 0
            for prune_threshold in range(self.__min_lexicon_freq):
                pruned_size += prune_threshold and freq_dict[prune_threshold] or 0
                if total - pruned_size <= self.__min_pruned_remain:
                    break
            
            pruned_model_param = deepcopy(model_param)
            pruned_model_param['lexicon'] = {k: v for k, v in model_param['lexicon'].items() if v > prune_threshold}
            remaining_morphs = [__map_key(_.split('_')) for _ in pruned_model_param['lexicon'].keys()]
            pruned_charset = set(''.join([morph for morph, _ in remaining_morphs]))
            self.__io.write_temp_file('pruned_model_param', pruned_model_param)
            self.__io.write_temp_file('remaining_morphs', remaining_morphs)
            log_wrapper("distributed.scheduler", 'Prune threshold: {}'.format(prune_threshold))
            log_wrapper("distributed.scheduler", 'Morphs remain: {}'.format(len(remaining_morphs)))
            log_wrapper("distributed.scheduler", 'Pruned charset: {}'.format(len(pruned_charset)))
        
        partition_with_arg = self.client.scatter([
            (i, self.__io.base_path) 
            for i in range(self.__num_partitions)])
        futures =  self.client.map(_map_segment, partition_with_arg)
        reduce_step = self.client.submit(_reduce_segment, futures)
        segmented_corpus, pruned_segmented_corpus = reduce_step.result()
        log_wrapper("distributed.scheduler", 'Final segmenting finished...')
        return segmented_corpus, pruned_segmented_corpus, pruned_model_param
    
    def __bulk_de_registration(self, model_param, last_iteration=False):
        log_wrapper("distributed.scheduler", 'Bulk de-registration started...')
        deregistered_morph = set()
        __map_key = lambda x: (x[0], int(x[1]))
        for i, (k, count) in enumerate(sorted(model_param['lexicon'].items(), key=lambda x: -x[1])):
            morph, state = __map_key(k.split('_'))
            r = random.random()
            # 17 States, 3 Prefixes, 3 Suffixes, 0 for beginning and 16 for end
            if (state <= self.__num_prefix or state > self.num_state - self.__num_suffix - 2) and \
                count < self.__affix_lbound and r < self.__bulk_prob:
                # Prefixes: 1 ~ 3 -> state <= num_prefix
                # Suffixes: 13 ~ 15 -> state > 17 - 3 - 2
                deregistered_morph.add((morph, state))
            elif self.__num_prefix < state <= self.num_state - self.__num_suffix - 2 and count > self.__stem_ubound and \
                r < self.__bulk_prob:
                # Stem: 4 ~ 12 -> num_prefix < state <= num_state - num_suffix - 2
                deregistered_morph.add((morph, state))
              
        model_param['deregistered_morph'] = deregistered_morph
        self.__io.write_temp_model_params(model_param)
        partition_with_arg = self.client.scatter([
            (i, self.__io.base_path, 0.0, 0.0) for i in range(self.__num_partitions)])
        futures =  self.client.map(_map_step, partition_with_arg)
        reduce_step = self.client.submit(
                    _reduce_step_wrapper(
                        self.num_state, self.__num_prefix, self.__num_suffix, 
                        self.__transition_ctrl, self.__has_word_boundary), 
                    futures)
        new_model_param, loss = reduce_step.result()
        self.__io.write_temp_model_params(new_model_param)
        log_wrapper("distributed.scheduler", 'Bulk de-registration finished...')
        log_wrapper("distributed.scheduler", 'Removed morphs: {}'.format(len(deregistered_morph)))
        return loss, new_model_param

    def train(self, max_iteration=10, bulk_dereg_every_n_epoch=0, save_corpus=False) -> BaseModel:
        """
        Train StateMorph model.
        
        Parameters
        ----------
        max_iteration: int
            Maximal number of iteration to train. Default is 10.
            Equivalent to distributed batch segmenting if set to 0.
            
        bulk_dereg_every_n_epoch: int
            Bulk de-registration every n epoch. Default is 0.
            
        save_corpus: bool
            Save segmented corpus to model binary. Default is False.
        
        Returns
        -------
        model: BaseModel
            Trained StateMorph model.
        
        """
        p_loss = -1
        count = 0
        model_param = self.__init_model_param
        temp = math.inf
        if self._final_temp> 0 and self._current_temp > 0 and self._alpha > 0:
            temp = math.ceil((math.log2(self._final_temp) - math.log2(self._current_temp)) / math.log2(self._alpha))        
        total_iteration = min(temp, max_iteration)
        log_wrapper("distributed.scheduler", 'Initial cost: {}'.format(self.__init_loss))
        for _ in range(total_iteration):
            self._current_temp = max(self._final_temp, self._current_temp * self._alpha)
            loss, model_param = self.__step(_, total_iteration)
            if random.random() < (math.exp(_/(total_iteration / 3.0)) - 1) / (math.exp(3) - 1) and \
                self.__bulk_prob > 0 and (self.__num_prefix > 0 or self.__num_suffix > 0) or \
                    bulk_dereg_every_n_epoch and _ % bulk_dereg_every_n_epoch == 0:
                loss, model_param = self.__bulk_de_registration(model_param, last_iteration=_ == total_iteration - 1)
            
            # Early stopping
            if abs(p_loss - loss) < self._delta and loss:
                count += 1
                if count == self._patience:
                    log_wrapper("distributed.scheduler", 'Early stopping...')
                    break
            elif self._current_temp < self._final_temp:
                break
            else:
                if p_loss > loss:
                    self.__checkpoint(BaseModel(model_param), _, loss)
                count = 0
                p_loss = loss
        
        segmented_corpus, pruned_segmented_corpus, pruned_model_param = self.__collect(model_param)
        new_model = BaseModel(model_param)
        final_cost = new_model.compute_encoding_cost()
        new_model.update_segmented_corpus(segmented_corpus, update_model=False)
        self.__checkpoint(new_model, 'FINAL', final_cost, save_corpus=save_corpus)
        if pruned_model_param and len(pruned_segmented_corpus):
            pruned_model = BaseModel(pruned_model_param)
            pruned_model.update_segmented_corpus(pruned_segmented_corpus, update_model=False)
            self.__checkpoint(pruned_model, 'PRUNED', final_cost, save_corpus=save_corpus)
            self.__io.dump_charset(pruned_model, 'pruned_charset.bin')
        # self.__io.remove_temp_files()
        return new_model