import math
import random
from collections import Counter
import numpy as np

class BaseModel(object):
    PRIOR = 0.5
    LOG_2 = math.log(2)
    HALF_LOG_2_PI = 0.5 * math.log(2.0 * math.pi)
    MORPH_SIZE = 8
    WORD_BOUNDARY = '▁'
    def __init__(self, model_param):
        '''
        # Suitable for multiprocessing
        # model_params = json.load(open('model_params.json', 'r', encoding='utf-8'))
        # segmenter = Segmenter(model_params=model_params)
        
        # Segment
        # segmenter.segment('aaltoenergiaa')
        # Output
        # {
            'morphs': list of morphs,
            'states': list of stats,
            'segment': list of morph-state tuples,
            'cost': cost
        }

        # Varify result
        # aaltoenergiaa
        # aalto 4 energia 11 a 8 	31.6421
        BaseModel.debug_segment('aaltoenergiaa', [('aalto', 4), ('energia', 11), ('a', 8)], 31.6421)
        '''
        self.num_state = 0 # Number of states
        self.lexicon = {} # Lexicon {(morph, state): freq}
        self.state_freq = {} # State frequency {state: freq}
        self.num_prefix = 0
        self.num_suffix = 0
        self.transition_freq = [] # Transition frequency {from_state: {to_state: freq}}
        self.transition_ctrl = {}
        self.segmented_corpus = []
        self.__cached_segment = {}
        self.__has_word_boundary = False
        self.__load_model_params(model_param)
        # double result = 0.0;
        # for(double x = count ; x < count+added ; x++){
        #     result += log2(x);
        # }
        # return result;
        
    
    def get_param_dict(self) -> dict:
        model_params = {
            'num_state': self.num_state,
            'num_prefix': self.num_prefix,
            'num_suffix': self.num_suffix,
            'lexicon': {'{}_{}'.format(*k): v for k, v in self.lexicon.items()},
            'state_freq': self.state_freq,
            'transition_freq': self.transition_freq,
            'transition_ctrl': self.transition_ctrl,
            'has_word_boundary': self.__has_word_boundary,
        }
        return model_params
        
    def summary(self) -> None:
        print('Has word boundary: {}'.format(self.__has_word_boundary))
        print('Number of states: {}'.format(self.num_state))
        print('Number of prefix states: {}'.format(self.num_prefix))
        print('Number of suffix states: {}'.format(self.num_suffix))
        print('Number of lexicon: {}'.format(len(self.lexicon)))
        print('Encoding cost: {}'.format(self.compute_encoding_cost()))
        print('Transition ctrl: {}'.format(self.transition_ctrl))
        print('State freq: {}'.format(sorted(self.state_freq.items(), key=lambda x: x[0])))
        print('Transition freq: ')
        for _ in self.transition_freq:
            print(_)

    def __load_model_params(self, model_params:dict):
        __map_key = lambda x: (x[0], int(x[1]))
        self.__has_word_boundary = model_params.get('has_word_boundary', False)
        self.num_state = model_params['num_state']
        self.lexicon = {__map_key(k.split('_')): v for k, v in model_params['lexicon'].items()}
        self.state_freq = {int(k): v for k, v in model_params['state_freq'].items()}
        self.transition_freq = model_params['transition_freq']
        self.num_prefix = model_params['num_prefix']
        self.num_suffix = model_params['num_suffix']
        self.transition_ctrl = model_params.get('transition_ctrl', {})
        self.__pre_charset = model_params.get('charset', set())
        self.__deregistered_morph = model_params.get('deregistered_morph', set())
        self.__update_counts()

    def __update_counts(self) -> None:
        self.__state_size = {k : 0 for k in range(1, self.num_state - 1)}
        self.__state2morph2freq = {k: {} for k in range(1, self.num_state - 1)}
        self.__morph2state2freq = {}
        __state2char2counts = {k : {} for k in range(1, self.num_state - 1)}
        __state_char_size = {k: 0 for k in range(1, self.num_state - 1)}
        self.__charset = {_ for _ in self.__pre_charset}
        for (morph, state), count in self.lexicon.items():
            self.__state2morph2freq[state][morph] = count
            if morph not in self.__morph2state2freq:
                self.__morph2state2freq[morph] = {}
            self.__morph2state2freq[morph][state] = count
            # lexicon.getMorphList(state).stream().map((Morphable m) -> m.getLength() + 1).reduce(Integer::sum).get();
            __state_char_size[state] += self.__get_morph_length(morph) + 1
            self.__state_size[state] += 1
            for char, _ in self.__get_morph_char_count(morph).items():
                __state2char2counts[state][char] = __state2char2counts[state].get(char, 0) + _ * count
                self.__charset.add(char)
        self.__num_bit_per_char = math.ceil(math.log(len(self.__charset) + 1) /  BaseModel.LOG_2)
        
        
        self.__transition_costs = [[self.__get_transition_cost(i, j) 
                                    for j in range(self.num_state)] 
                                    for i in range(self.num_state)]
        # double result = 0.0;
        # for(double x = count ; x < count+added ; x++){
        #     result += log2(x);
        # }
        # return result;
        __compute_log_gamma_change = lambda count, added: np.log2(np.arange(count, count + added, 1)).sum()
        
        # double sumWithPriors = 0.0;
        # if (!lexicon.getMorphList(state).isEmpty()) {
        #     sumWithPriors = lexicon.getMorphList(state).stream().map((Morphable m) -> m.getLength() + 1).reduce(Integer::sum).get();
        # }
        # sumWithPriors += (constants.customCharsetIndex.length + 1) * constants.getPrior();
        # double addedToSum = constants.allMorphs[morphId].length + 1;
        # double posPart = AmorphousMath.computeLogGammaChange(sumWithPriors, addedToSum);        
        self.__add_morph_pos = {
            (state, morph_size): __compute_log_gamma_change(
                    __state_char_size[state] + (len(self.__charset) + 1) * BaseModel.PRIOR , morph_size + 1)
            for state in range(1, self.num_state - 1)
            for morph_size in range(BaseModel.MORPH_SIZE + 1)
        }
                
        
        # double negPart = 0.0;
        # for (Byte b : mapForMorph.keySet()) {
        #     double countWithPrior = currentCharCounts.getOrDefault(b, 2) + constants.getPrior();
        #     double added = mapForMorph.get(b);
        #     negPart += AmorphousMath.computeLogGammaChange(countWithPrior, added);
        # }
        self.__add_morph_neg_1 = {
            (state, char, count): __compute_log_gamma_change(
                __state2char2counts[state].get(char, 2) + BaseModel.PRIOR, count)
            for state in range(1, self.num_state - 1)
            for char in self.__charset
            for count in range(BaseModel.MORPH_SIZE + 1)
        }
        # double countWithPrior = lexicon.getMorphList(state).size() + constants.getPrior();
        # double added = 1;
        # negPart += AmorphousMath.computeLogGammaChange(countWithPrior, added);
        self.__add_morph_neg_2 = {
            state: __compute_log_gamma_change(self.__state_size[state] + BaseModel.PRIOR, 1)
            for state in range(1, self.num_state - 1)
        }

    def update_segmented_corpus(self, segmented_corpus, update_model=True, build_cache=False) -> None:
        self.segmented_corpus = segmented_corpus
        if build_cache:
            self.__cached_segment = {
                ''.join([morph for morph, _ in segment]).lstrip(BaseModel.WORD_BOUNDARY): (segment, cost)
                for (segment, cost) in segmented_corpus
            }
        if update_model:
            self.__update_model()
            self.__update_counts()
    
    def train_step(self, corpus=[], temperature=0, is_final=False) -> tuple:
        segmented_corpus = []
        for segment, _ in self.segmented_corpus:
            word = ''.join([morph for morph, _ in segment]).lstrip(BaseModel.WORD_BOUNDARY)
            new_segment, new_cost = self.__search(word, temperature=temperature, is_training=not is_final)
            if new_cost != math.inf:
                segmented_corpus.append((new_segment, new_cost))
            else:
                segmented_corpus.append((segment, _))
        for word in corpus:
            new_segment, new_cost = self.__search(word, temperature=temperature, is_training=not is_final)
            if new_cost != math.inf:
                segmented_corpus.append((new_segment, new_cost))
        segmented_corpus = [(segment, cost) 
            for segment, cost in segmented_corpus 
            if cost > 0 and all(k not in self.__deregistered_morph for k in segment)
        ]
        self.update_segmented_corpus(segmented_corpus)
        return self.get_param_dict(), segmented_corpus
    
    def __update_model(self) -> None:
        lexicon = {}
        state_freq = {k : 0 for k in range(self.num_state)}
        end_state = self.num_state - 1
        transition_freq_dict = {}
        for segment, cost in self.segmented_corpus:
            p_state = 0
            state_freq[0] += 1
            for morph, state in segment:
                if (morph, state) not in lexicon:
                    lexicon[(morph, state)] = 0
                lexicon[(morph, state)] += 1
                if p_state not in transition_freq_dict:
                    transition_freq_dict[p_state] = {}
                if state not in transition_freq_dict[p_state]:
                    transition_freq_dict[p_state][state] = 0
                transition_freq_dict[p_state][state] += 1
                state_freq[state] += 1
                p_state = state
            state_freq[end_state] += 1
            if state not in transition_freq_dict:
                transition_freq_dict[state] = {}
            transition_freq_dict[state][end_state] = transition_freq_dict[state].get(end_state, 0) + 1            

        self.lexicon = lexicon
        self.state_freq = state_freq
        self.transition_freq = [[transition_freq_dict.get(i, {}).get(j, 0) 
                            for j in range(self.num_state)] 
                           for i in range(self.num_state)]
        
    
    def compute_encoding_cost(self) -> float:
        # PrequentialCost
        transition_encoding_cost = self.__get_transition_encoding_cost()
        lexicon_encoding_cost, emissions_encoding_cost = self.__get_lexicon_and_emission_encoding_cost()
        return lexicon_encoding_cost + emissions_encoding_cost + transition_encoding_cost

    def segment(self, word: str) -> tuple:
        assert set(word).issubset(self.__charset), f'Word {word} contains characters not in charset'
        try:
            return self.__cached_segment[word]
        except KeyError:
            return self.__search(word)
    
    def __search(self, word: str, temperature=0, is_training=False) -> tuple :
        to_be_segmented = f'${word}#'
        word_boundary_size = 0
        if self.__has_word_boundary:
            to_be_segmented = f'${BaseModel.WORD_BOUNDARY}{word}#'
            word_boundary_size = len(BaseModel.WORD_BOUNDARY)
        dp_matrix = [[{
            'state': state, 
            'char_index': char_index, 
            'cost': 0 if char_index ==0 and state == 0 else math.inf, 
            'previous': None, 
            'morph': to_be_segmented[char_index]
        } for char_index in range(len(word) + 2 + word_boundary_size)] for state in range(self.num_state)]
        dp_matrix[0][0]['cost'] = 0
        for char_idx in range(1 + word_boundary_size, len(word) + 2 + word_boundary_size):
            for state in range(1, self.num_state):
                searching_middle = char_idx != len(word) + 1 + word_boundary_size and state != self.num_state - 1
                searching_end = char_idx == len(word) + 1 + word_boundary_size and state == self.num_state - 1
                if searching_middle or searching_end:
                    current_cell = dp_matrix[state][char_idx]
                    search_space = [
                        cell 
                        for chars in dp_matrix[: -1]
                        for cell in chars[max(char_idx - BaseModel.MORPH_SIZE, 0): char_idx]
                        if not self.__has_word_boundary or \
                            cell['char_index'] == 0 or cell['char_index'] > len(BaseModel.WORD_BOUNDARY)
                    ]
                    
                    if searching_end:
                        search_space = [cell for chars in dp_matrix[1: -1] for cell in chars[-2:-1]]
                    
                    
                    costs = []
                    for idx, previous_cell in enumerate(search_space):
                        morph = '#'
                        cost = previous_cell['cost']
                        cost += self.__transition_costs[previous_cell['state']][current_cell['state']]
                        if not searching_end:
                            morph = to_be_segmented[previous_cell['char_index'] + 1: current_cell['char_index'] + 1]
                            cost += self.__get_emission_cost(morph, current_cell['state'], is_training=is_training)
                        costs.append((idx, cost))
                    candidates = sorted(costs, key=lambda x: x[1])
                    window_size = int(max(min(temperature / 100 * len(candidates), len(candidates)), 1))
                    candidates = [candidates[0]] + [_ for _ in candidates[1: window_size] if not math.isinf(_[1])]
                    idx, cost = random.choice(candidates)
                    current_cell['previous'] = search_space[idx]
                    current_cell['cost'] = cost
                    if not searching_end:
                        current_cell['morph'] = to_be_segmented[search_space[idx]['char_index'] + 1: 
                                                                current_cell['char_index'] + 1]
        p = dp_matrix[-1][-1]
        cost = p['cost']
        reversed_path = []
        while p['previous'] is not None:
            reversed_path.append(p)
            p = p['previous']
        segment = list(map(lambda x: (x['morph'], x['state']), reversed(reversed_path)))[:-1]
        return segment, cost

    def debug_dp_matrix(self, word, dp_matrix, segment) -> None:
        to_be_segmented = '$' + word + '#'
        print('-------------------')
        print(','.join(to_be_segmented))
        for row in dp_matrix:
            print(','.join(['{}:{:.3f}:{:.3f}'.format(
                col['morph'], col['cost'],( col['previous'] or {}).get('cost', math.inf))  for col in row]))
        print('-------------------')
        print(segment)

    def debug_segment(self, word:str, expected_segment: list, expected_cost: float) -> None:
        segment, cost = self.search(word)
        print('If same segment', len(segment)== len(expected_segment) and all(i==j for i, j in zip(segment, expected_segment)))
        print('Cost prec error:', float(format((cost - expected_cost) / expected_cost, '.5f')), '%')
    

    def __get_emission_cost(self, morph: str, state: int, is_training=False) -> float:
        if state == self.num_state - 1:
            return 0
        try:
            state_dict = self.__morph2state2freq[morph]
        except KeyError:
            state_dict = {}
        if state not in state_dict and is_training:
            # morph is not yet emitted from this class: add it:
            # We are not sure what this means. It seems  to produce a reasonable freq, estimate in marginal cases
            # Roman,: 5.0 could be  the closest power of two to size of alphabet
            # int morphLength = constants.allMorphs[morphId].length;
            # cost = - AmorphousMath.log2(constants.getPrior() / (s.classFrequency() + (s.classSize() + 1.0) * constants.getPrior()));
            cost = - math.log2(BaseModel.PRIOR / 
                               (self.state_freq[state] + (self.__state_size[state] + 1) * BaseModel.PRIOR))
            # cost += costOfAddingMorphToClass(state, morph)
            
            pos_part = self.__add_morph_pos[(state, self.__get_morph_length(morph))]
            neg_part = sum(map(lambda x: self.__add_morph_neg_1[(state, x[0], x[1])], self.__get_morph_char_count(morph).items()))
            neg_part += self.__add_morph_neg_2[state]
            
            cost += pos_part - neg_part
        elif state not in state_dict and not is_training:
            cost = math.inf
        else:
            # d = morph.getFrequency() + constants.getPrior();
            d = state_dict[state] + BaseModel.PRIOR
            #cost = - AmorphousMath.log2(d / (s.classFrequency() + s.classSize()*constants.getPrior()));
            cost =  - math.log2(d / (self.state_freq[state] + self.__state_size[state] * BaseModel.PRIOR))
        return cost


    def __get_transition_cost(self, state_a: int, state_b: int) -> float:
        try:
            transition_ctrl = self.transition_ctrl[(state_a, state_b)]
        except KeyError:
            transition_ctrl = 1
        if transition_ctrl:
            cost = - math.log2((self.transition_freq[state_a][state_b] + BaseModel.PRIOR) / 
                           (self.state_freq[state_a] + (self.num_state - 2) * BaseModel.PRIOR))
        else:
            cost = math.inf
        return cost


    def __get_transition_encoding_cost(self) -> float:
        length = 0
        
        for cfrom in range(self.num_state - 1):
            freq_sum = 0
            sum_of_priors = 0
            state_to_start = 1
            state_to_end = self.num_state - 1
            for cto in range(state_to_start, state_to_end + 1):
                if not cfrom and cto == self.num_state - 1:
                    continue
                freq_sum += self.transition_freq[cfrom][cto]
                sum_of_priors += BaseModel.PRIOR
                length -= math.lgamma(self.transition_freq[cfrom][cto] + BaseModel.PRIOR) / BaseModel.LOG_2
                length += math.lgamma(BaseModel.PRIOR) / BaseModel.LOG_2
            length += math.lgamma(max(freq_sum, 1)) / BaseModel.LOG_2
            length -= math.lgamma(sum_of_priors) / BaseModel.LOG_2
        return length
    
    
    def __get_lexicon_and_emission_encoding_cost(self) -> float:
        
        # double lexiconLength = 0;
        lexicon_length = 0
        # double emissionsLength = 0;
        emissions_length = 0
        
        # for (int c = START_STATE + 1; c < END_STATE; c++) {
        #     morphs = this.lexicon.getMorphList(c);
        #     classSize = this.lexicon.getClassSize(c);
        #     int sumCounts = 0;
        #     if (classSize > 0) {
        #         for (Morphable m : morphs) {
        #             lexiconLength += constants.getNumBitsForOneSymbol() * (m.getLength() + 1);                    
        #             emissionsLength -= AmorphousMath.log2Gamma(m.getCounter() + constants.getPrior());        
        #             emissionsLength += AmorphousMath.log2Gamma(constants.getPrior());
        #             sumCounts += m.getCounter();
        #         }
                
        #         emissionsLength += AmorphousMath.log2Gamma(sumCounts + classSize*constants.getPrior());
        #         emissionsLength -= AmorphousMath.log2Gamma(classSize*constants.getPrior());
                
        #         //Subract the number of bits we have saved from transmitting morphs in a specific order :
        #         emissionsLength -= AmorphousMath.log2Gamma(classSize + 1);
        #     }
            
            
        # }
        for state in range(1, self.num_state - 1):
            morphs = self.__state2morph2freq[state].keys()
            class_size = self.__state_size[state]
            sum_counts = 0
            if class_size > 0:
                for morph in morphs:
                    lexicon_length += self.__num_bit_per_char * (self.__get_morph_length(morph) + 1)
                    emissions_length -= math.lgamma(self.__state2morph2freq[state][morph] + BaseModel.PRIOR)
                    emissions_length += math.lgamma(BaseModel.PRIOR)
                    sum_counts += self.__state2morph2freq[state][morph]
                
                emissions_length += math.lgamma(sum_counts + class_size * BaseModel.PRIOR)
                emissions_length -= math.lgamma(class_size * BaseModel.PRIOR)
                
                emissions_length -= math.lgamma(class_size + 1)
        
        return lexicon_length, emissions_length
    
    def __get_morph_length(self, morph: str) -> int:
        """
        Returns the length of the morph, excluding the word boundary if present.
        """
        morph_length = len(morph)
        if self.__has_word_boundary and morph.startswith(BaseModel.WORD_BOUNDARY):
            morph_length -= len(BaseModel.WORD_BOUNDARY)  # Exclude word boundary character from length
        return morph_length

    def __get_morph_char_count(self, morph: str) -> Counter:
        """
        Returns a dictionary with character counts in the morph, excluding the word boundary if present.
        """
        temp = morph
        if self.__has_word_boundary and morph.startswith(BaseModel.WORD_BOUNDARY):
            temp = morph[len(BaseModel.WORD_BOUNDARY):]  # Exclude word boundary character
        return Counter(temp)