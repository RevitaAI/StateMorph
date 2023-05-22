import math
import random
from collections import Counter
import numpy as np

class BaseModel(object):
    PRIOR = 0.5
    LOG_2 = math.log(2)
    HALF_LOG_2_PI = 0.5 * math.log(2.0 * math.pi)
    MORPH_SIZE = 8
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
        self.__state_size = {} # State size {state: morph count}
        self.__morph2state2freq = {} # Morph dictionary {morph: {state: freq}}
        self.__state2char2counts = {} # State char counts {state: {char: freq}}
        self.transition_freq = [] # Transition frequency {from_state: {to_state: freq}}
        self.lexicon_costs = []
        self.transition_costs = []
        self.segmented_corpus = []
        self.__load_model_params(model_param)
        # double result = 0.0;
        # for(double x = count ; x < count+added ; x++){
        #     result += log2(x);
        # }
        # return result;
        
    
    def get_param_dict(self):
        model_params = {
            'num_state': self.num_state,
            'lexicon': {'{}_{}'.format(*k): v for k, v in self.lexicon.items()},
            'state_freq': self.state_freq,
            'transition_freq': self.transition_freq,
        }
        return model_params
        
    
    def __load_model_params(self, model_params:dict):
        __map_key = lambda x: (x[0], int(x[1]))
        self.num_state = model_params['num_state']
        self.lexicon = {__map_key(k.split('_')): v for k, v in model_params['lexicon'].items()}
        self.state_freq = {int(k): v for k, v in model_params['state_freq'].items()}
        self.transition_freq = model_params['transition_freq']
        self.update_counts()

    def update_counts(self):
        self.__state_size = {k : 0 for k in range(self.num_state + 2)}
        self.__state2char2counts = {k : {} for k in range(self.num_state + 2)}
        self.__state2morph2freq = {k: {} for k in range(1, self.num_state - 1)}
        self.__morph2state2freq = {}
        self.__state_char_size = {k: 0 for k in range(self.num_state)}
        self.__charset = set()
        for (morph, state), count in self.lexicon.items():
            self.__state2morph2freq[state][morph] = count
            if morph not in self.__morph2state2freq:
                self.__morph2state2freq[morph] = {}
            self.__morph2state2freq[morph][state] = count
            # lexicon.getMorphList(state).stream().map((Morphable m) -> m.getLength() + 1).reduce(Integer::sum).get();
            self.__state_char_size[state] += len(morph) + 1
            self.__state_size[state] += 1
            for char, _ in Counter(morph).items():
                self.__state2char2counts[state][char] = self.__state2char2counts[state].get(char, 0) + _ * count
                self.__charset.add(char)
            
        
        
        self.transition_costs = [[self.__get_transition_cost(i, j) 
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
                    self.__state_char_size[state] + (len(self.__charset) + 1) * BaseModel.PRIOR , morph_size + 1)
            for state in range(1, self.num_state)
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
                self.__state2char2counts[state].get(char, 2) + BaseModel.PRIOR, count)
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

        
        
    def update_segmented_corpus(self, segmented_corpus, update_model=True):
        self.segmented_corpus = segmented_corpus
        if update_model:
            self.update_model()
            self.update_counts()
    
    def train_step(self, corpus=[], temperature=0):
        segmented_corpus = []
        for segment, _ in self.segmented_corpus:
            word = ''.join([morph for morph, _ in segment])
            new_segment, new_cost = self.__search(word, temperature=temperature)
            if new_cost != math.inf:
                segmented_corpus.append((new_segment, new_cost))
            else:
                segmented_corpus.append((segment, _))
        for word in corpus:
            new_segment, new_cost = self.__search(word, temperature=temperature)
            if new_cost != math.inf:
                segmented_corpus.append((new_segment, new_cost))
        self.update_segmented_corpus([_ for _ in segmented_corpus if _[1] > 0])
        return self.get_param_dict(), segmented_corpus
    
    def update_model(self):
        lexicon = {}
        state_freq = {k : 0 for k in range(self.num_state + 2)}
        transition_freq_dict = {}
        for segment, _ in self.segmented_corpus:
            p_state = 0
            state_freq[0] = state_freq.get(0, 0) + 1
            for morph, state in segment:
                if (morph, state) not in lexicon:
                    lexicon[(morph, state)] = 0
                lexicon[(morph, state)] += 1
                if p_state not in transition_freq_dict:
                    transition_freq_dict[p_state] = {}
                if state not in transition_freq_dict[p_state]:
                    transition_freq_dict[p_state][state] = 0
                transition_freq_dict[p_state][state] += 1
                state_freq[state] = state_freq.get(state, 0) + 1
                p_state = state

        end_state = self.num_state - 1
        for segment, _ in self.segmented_corpus:
            state = segment[-1][1]
            state_freq[end_state] = state_freq.get(end_state, 0) + 1
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
        cost = 0
        cost += self.__get_transition_encoding_cost()
        cost += sum([self.__get_lexicon_cost(_) for _ in range(1, self.num_state)])
        cost += self.__get_emission_encoding_cost()
        return cost

    def __search(self, word: str, temperature=0) -> tuple :
        to_be_segmented = '$' + word + '#'
        dp_matrix = [[{
            'state': state, 
            'char_index': char_index, 
            'cost': 0 if char_index ==0 and state == 0 else math.inf, 
            'previous': None, 
            'morph': to_be_segmented[char_index]
        } for char_index in range(len(word) + 2)] for state in range(self.num_state)]
        dp_matrix[0][0]['cost'] = 0
        for state in range(1, self.num_state):
            for char_idx in range(1, len(word) + 2):
                searching_middle = char_idx != len(word) + 1 and state != self.num_state - 1
                searching_end = char_idx == len(word) + 1 and state == self.num_state - 1
                if searching_middle or searching_end:
                    current_cell = dp_matrix[state][char_idx]
                    search_space = [cell 
                                    for chars in dp_matrix[int(searching_end): -1] 
                                    for cell in chars[max(char_idx - BaseModel.MORPH_SIZE, int(searching_end)): char_idx]]
                    costs = []
                    for idx, previous_cell in enumerate(search_space):
                        morph = to_be_segmented[previous_cell['char_index'] + 1: current_cell['char_index'] + 1]
                        cost = previous_cell['cost']
                        cost += self.transition_costs[previous_cell['state']][current_cell['state']]
                        cost += self.__get_emission_cost(morph, current_cell['state'])
                        costs.append((idx, cost))
                    candidates = sorted(costs, key=lambda x: x[1])
                    window_size = int(max(min(temperature / 100 * len(candidates), len(candidates)), 1))
                    candidates = [candidates[0]] + [_ for _ in candidates[1: window_size] if not math.isinf(_[1])]
                    idx, cost = random.choice(candidates)
                    current_cell['previous'] = search_space[idx]
                    current_cell['cost'] = cost
                    if not searching_end:
                        current_cell['morph'] = to_be_segmented[search_space[idx]['char_index'] + 1: current_cell['char_index'] + 1]
        p = dp_matrix[-1][-1]
        cost = p['cost']
        reversed_path = []
        while p['previous'] is not None:
            reversed_path.append(p)
            p = p['previous']
        segment = list(map(lambda x: (x['morph'], x['state']), reversed(reversed_path)))[:-1]
        return segment, cost

    def debug_dp_matrix(self, word, dp_matrix, segment):
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
    

    def __get_emission_cost(self, morph: str, state: int) -> float:
        if state == self.num_state - 1:
            return 0
        state_dict = self.__morph2state2freq.get(morph, {})
        if state not in state_dict:
            # morph is not yet emitted from this class: add it:
            # We are not sure what this means. It seems  to produce a reasonable freq, estimate in marginal cases
            # Roman,: 5.0 could be  the closest power of two to size of alphabet
            # int morphLength = constants.allMorphs[morphId].length;
            # cost = - AmorphousMath.log2(constants.getPrior() / (s.classFrequency() + (s.classSize() + 1.0) * constants.getPrior()));
            cost = - math.log2(BaseModel.PRIOR / 
                               (self.state_freq.get(state, 0) + (self.__state_size[state] + 1) * BaseModel.PRIOR))
            # cost += costOfAddingMorphToClass(state, morph)
            
            pos_part = self.__add_morph_pos[(state, len(morph))]
            neg_part = sum(map(lambda x: self.__add_morph_neg_1[(state, x[0], x[1])], Counter(morph).items()))
            neg_part += self.__add_morph_neg_2[state]
            
            cost += pos_part - neg_part
        else:
            # d = morph.getFrequency() + constants.getPrior();
            d = state_dict[state] + BaseModel.PRIOR
            #cost = - AmorphousMath.log2(d / (s.classFrequency() + s.classSize()*constants.getPrior()));
            cost =  - math.log2(d / (self.state_freq[state] + self.__state_size[state] * BaseModel.PRIOR))
        return cost


    def __get_transition_cost(self, state_a: int, state_b: int) -> float:
        cost = - math.log2((self.transition_freq[state_a][state_b] + BaseModel.PRIOR) / 
                           (self.state_freq.get(state_a, 0) + (self.num_state - 2)* BaseModel.PRIOR))
        return cost

    def __get_simple_lexicon_cost(self, morph: str) -> float:
        return len(morph) + 1 + math.log(len(self.__charset) + 2)


    def __get_lexicon_cost(self, state_id: int) -> float:
        #int classSize = lexicon.getMorphList(state).size();
        #double classLexCost = computePrequentialCostForMap(classCount);
        if not state_id or state_id == self.num_state - 1:
            state_lex_cost = self.__get_cost([])
        else:
            state_lex_cost = self.__get_cost(list(self.__state2char2counts[state_id].values()))
        #we add one because Elias coding 
        # works only for positive integers but we deal with nonnegative integers.
        #double costOfCodingSize = amorphous.math.AmorphousMath.computeEliasOmegaLength(classSize + 1); 
        cost_of_coding_size = self.__compute_elias_omega_length(self.__state_size[state_id] + 1)
        return cost_of_coding_size + state_lex_cost


    def __get_cost(self, counts: list) -> float:
        if not len(counts):
            return 0
        cost = 0
        sumOfEvents = 0
        sumOfPriors = 0
        for d in counts:
            cost -= math.lgamma(d + BaseModel.PRIOR) / BaseModel.LOG_2
            cost += math.lgamma(BaseModel.PRIOR) / BaseModel.LOG_2
            sumOfEvents += d + BaseModel.PRIOR
            sumOfPriors += BaseModel.PRIOR

        cost += math.lgamma(sumOfEvents) / BaseModel.LOG_2
        cost -= math.lgamma(sumOfPriors) / BaseModel.LOG_2
        return cost


    def __compute_elias_omega_length(self, x:int) -> float:
        length = 0
        lambda_i_number = math.floor(math.log2(x))
        while lambda_i_number > 0:
            length += lambda_i_number + 1
            lambda_i_number = math.floor(math.log2(lambda_i_number))
        return length

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
            length += math.lgamma(freq_sum) / BaseModel.LOG_2
            length -= math.lgamma(sum_of_priors) / BaseModel.LOG_2
        return length
    
    def __get_emission_encoding_cost(self):
        emit_cost = 0
        for state in range(1, self.num_state - 1):
            counts = list(self.__state2morph2freq[state].values())
            emit_cost += self.__get_cost(counts)
        return emit_cost