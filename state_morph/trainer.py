
from .core import BaseModel
from .utils import _map_step, _reduce_step
import random
from statistics import mean
from multiprocessing import Pool

class StateMorphTrainer(object):
    def __init__(self, num_state, delta=1e-6, patience=10, init_temp=100, final_temp=1e-4, alpha=0.95) -> None:
        self.num_state = num_state
        self.__delta = delta
        self.__patience = patience
        self.__final_temp = final_temp
        self.__alpha = alpha
        self.__current_temp = init_temp
    
    def load_raw_corpus(self, corpus_file, **kwargs) -> None:
        """Load corpus to state morphology model."""
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.read().splitlines()
            segmented_corpus = self.__random_segment(corpus)
            model_params = {
                'morph_dict':  {},
                'state_freq': {},
                'state_size': {},
                'state_char_counts': {},
                'transition_freq': [],
            }
            model = BaseModel(model_params, **kwargs)
            model.update_segmented_corpus(segmented_corpus)
            self.__base_model = model
    
    def __random_segment(self, corpus) -> list:
        segmented_corpus = []
        for word in corpus:
            segment = []
            if len(word) > 1:
                j = 0
                for i in random.sample(list(range(1, len(word))), random.randint(1, len(word) - 1)):
                    morph = word[j:i]
                    if len(morph) >= 1:
                        segment.append((morph, random.randint(1, self.num_state)))
                        j = i
                morph = word[j:]
                if len(morph) >= 1:
                    segment.append((morph, random.randint(1, self.num_state)))
            else:
                segment.append((word, random.randint(1, self.num_state)))
            segmented_corpus.append((segment, 0))
        return segmented_corpus
             
    def __merge_morph(self, segmented_corpus) -> list:
        corpus = []
        for segment, _ in segmented_corpus:
            word = ''.join([morph for morph, _ in segment])
            corpus.append(word)
        return corpus
    
    
    def train(self, iteration=10, num_processes=5) -> BaseModel:
        """Train state morphology model."""
        model_param = self.__base_model.get_param_dict()
        segmented_corpus = self.__base_model.segmented_corpus
        p_loss = -1
        count = 0
        for _ in range(iteration):
            print('Iteration:', _, 'Temperature:', self.__current_temp)
            partition_size = len(segmented_corpus) // num_processes
            partitions = [segmented_corpus[i:i+partition_size]
                         for i in range(0, len(segmented_corpus), partition_size)]
            temp = partitions[:-1]
            temp[-1] += partitions[-1]
            partitions = temp 
            partition = [(_, model_param, partition) for _, partition in enumerate(partitions)]
            
            total_model_param = {
                'morph_dict': {},
                'state_freq': {},
                'state_size': {},
                'state_char_counts': {},
                'transition_freq': [],
            }
            total_segmented_corpus = []
            with Pool(num_processes) as p:
                for model_param, segmented_corpus in p.starmap(_map_step, partition):
                    _reduce_step(total_model_param, total_segmented_corpus, model_param, segmented_corpus)
                p.close()
                p.join()
            print('Reduce step finished...')
            model_param = total_model_param
            segmented_corpus = total_segmented_corpus
            loss = mean([cost for _, cost in segmented_corpus if cost > 0])
            print('Iteration: {}, Cost: {}'.format(_, loss))
            
            # Early stopping
            if abs(p_loss - loss) < self.__delta:
                count += 1
                if count == self.__patience:
                    print('Early stopping...')
                    break
            else:
                count = 0
                p_loss = loss
                
            random.shuffle(segmented_corpus)
            i = int(len(segmented_corpus) * self.__current_temp)
            segmented_corpus = self.__random_segment(self.__merge_morph(segmented_corpus[:i])) + segmented_corpus[i:]
            self.__current_temp = max(self.__final_temp, self.__current_temp * self.__alpha)
            
        new_model = BaseModel(model_param)
        new_model.update_segmented_corpus(segmented_corpus, update_model=False)
        return new_model