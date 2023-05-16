import math
from copy import deepcopy

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
        
        self.morph_dict = {} # Morph dictionary {morph: {state: freq}}
        self.state_freq = {} # State frequency {state: freq}
        self.state_size = {} # State size {state: morph count}
        self.state_char_counts = {} # State char counts {state: {char: freq}}
        self.transition_freq = [] # Transition frequency {from_state: {to_state: freq}}
        self.lexicon_costs = []
        self.transition_costs = []
        self.segmented_corpus = []
        self.__load_model_params(model_param)
    
    def get_param_dict(self):
        model_params = {
            'morph_dict': self.morph_dict,
            'state_freq': self.state_freq,
            'state_size': self.state_size,
            'state_char_counts': self.state_char_counts,
            'transition_freq': self.transition_freq,
        }
        return model_params
        
    
    def __load_model_params(self, model_params:dict):
        self.morph_dict = {k: {int(vk): vv for vk, vv in v.items()} for k, v in model_params['morph_dict'].items()}
        self.state_freq = {int(k): v for k, v in model_params['state_freq'].items()}
        self.state_size = {int(k): v for k, v in model_params['state_size'].items()}
        self.state_char_counts = {int(k): v for k, v in model_params['state_char_counts'].items()}
        self.transition_freq = model_params['transition_freq']
        self.update_costs()

    def update_costs(self):
        self.state_num = len(self.state_freq)
        self.charset = {_ for morph in self.morph_dict.keys() for _ in morph}
        self.lexicon_costs = [self.__get_lexicon_cost(_) for _ in range(1, self.state_num)]
        self.transition_costs = [[self.__get_transition_cost(i, j) 
                                  for j in range(self.state_num)] 
                                 for i in range(self.state_num)]
        
    def update_segmented_corpus(self, segmented_corpus, update_model=True):
        self.segmented_corpus = segmented_corpus
        if update_model:
            self.update_model()
    
    def train_step(self, corpus=[]):
        segmented_corpus = []
        for segment, _ in self.segmented_corpus:
            word = ''.join([morph for morph, _ in segment])
            new_segment, new_cost = self.search(word)
            if new_cost != math.inf:
                segmented_corpus.append((new_segment, new_cost))
            else:
                segmented_corpus.append((segment, _))
        for word in corpus:
            new_segment, new_cost = self.search(word)
            if new_cost != math.inf:
                segmented_corpus.append((new_segment, new_cost))
        self.update_segmented_corpus([_ for _ in segmented_corpus if _[1] > 0])
        return self.get_param_dict(), segmented_corpus
    
    def update_model(self):
        morph_dict = {}
        state_freq = {}
        state_size = {}
        state_char_counts = {}
        transition_freq_dict = {}
        __state_morph_set = {}
        for segment, _ in self.segmented_corpus:
            p_state = 0
            state_freq[0] = state_freq.get(0, 0) + 1
            for morph, state in segment:
                __state_morph_set[state] = __state_morph_set.get(state, set())
                __state_morph_set[state].add(morph)
                if morph not in morph_dict:
                    morph_dict[morph] = {}
                if state not in morph_dict[morph]:
                    morph_dict[morph][state] = 0
                morph_dict[morph][state] += 1
                if p_state not in transition_freq_dict:
                    transition_freq_dict[p_state] = {}
                if state not in transition_freq_dict[p_state]:
                    transition_freq_dict[p_state][state] = 0
                transition_freq_dict[p_state][state] += 1
                for c in morph:
                    if state not in state_char_counts:
                        state_char_counts[state] = {}
                    if c not in state_char_counts[state]:
                        state_char_counts[state][c] = 0
                    state_char_counts[state][c] += 1
                state_freq[state] = state_freq.get(state, 0) + 1
                p_state = state

        end_state = len(state_freq)
        for segment, _ in self.segmented_corpus:
            state = segment[-1][1]
            state_freq[end_state] = state_freq.get(end_state, 0) + 1
            if state not in transition_freq_dict:
                transition_freq_dict[state] = {}
            transition_freq_dict[state][end_state] = transition_freq_dict[state].get(end_state, 0) + 1
        
        state_size = {0: 0, end_state: 0}
        state_size.update({k: len(v) for k, v in __state_morph_set.items()})
        self.morph_dict = morph_dict
        self.state_freq = state_freq
        self.state_size = state_size
        self.state_char_counts = state_char_counts
        self.transition_freq = [[transition_freq_dict.get(i, {}).get(j, 0) 
                            for j in range(end_state + 1)] 
                           for i in range(end_state + 1)]
        self.update_costs()
    

    def compute_cost(self, segment: list) -> float:
        # self.__get_lexicon_cost(0) + self.__get_transition_costs(0, 1) + self.__get_emission_cost(0, '')
        cost = 0
        morphs = list(map(lambda x: x[0], segment))
        states = list(map(lambda x: x[1], segment))
        # cost += sum([self.lexicon_costs[_] for _ in states])
        # cost += sum([self.__get_simple_lexicon_cost(_) for _ in morphs])
        transitions = [(states[_], states[_ + 1]) for _ in range(len(states) - 1)]
        transitions.insert(0, (0, states[0]))
        transition_cost = sum([self.transition_costs[a][b] for a, b in transitions])
        emisstion_cost = sum([self.__get_emission_cost(morph, state) for morph, state, in segment])
        cost += transition_cost
        cost += emisstion_cost
        return cost

    def search(self, word: str) -> tuple :
        dp_matrix = [[{'state': state, 'char_index': char_index, 'cost': math.inf, 'previous': None, 'morph': ''} 
                      for state in range(self.state_num)] 
                    for char_index in range(len(word) + 2)]
        dp_matrix[0][0]['cost'] = 0
        for row in range(1, len(word) + 2):
            for col in range(1, self.state_num):
                if (row != len(word) + 1 and col != self.state_num - 1) or \
                   (row == len(word) + 1 and col == self.state_num - 1):
                    current_cell = dp_matrix[row][col]
                    search_space = [cell for rows in dp_matrix[max(row - BaseModel.MORPH_SIZE, 0): row] for cell in rows[: -1]]
                    costs = []
                    for previous_cell in search_space:
                        morph = word[previous_cell['char_index']: current_cell['char_index']]
                        cost = previous_cell['cost']
                        if not math.isinf(cost**2):
                            cost += self.transition_costs[previous_cell['state']][current_cell['state']]
                            cost += self.__get_emission_cost(morph, current_cell['state'])
                            if math.isinf(cost):
                                cost = cost ** 2
                        costs.append(cost)
                    idx, cost = min(enumerate(costs), key=lambda x: x[1])
                    current_cell['previous'] = search_space[idx]
                    current_cell['cost'] = cost
                    current_cell['morph'] = word[search_space[idx]['char_index']: current_cell['char_index']]
        p = dp_matrix[-1][-1]
        cost = p['cost']
        reversed_path = []
        while p['previous'] is not None:
            reversed_path.append(p)
            p = p['previous']
        segment = list(map(lambda x: (x['morph'], x['state']), reversed(reversed_path)))[:-1]
        return segment, cost

    def debug_segment(self, word:str, expected_segment: list, expected_cost: float) -> None:
        segment, cost = self.search(word)
        print('If same segment', len(segment)== len(expected_segment) and all(i==j for i, j in zip(segment, expected_segment)))
        print('Cost prec error:', float(format((cost - expected_cost) / expected_cost, '.5f')), '%')


    def __get_emission_cost(self, morph: str, state: int) -> float:
        if morph == '' and state == self.state_num - 1:
            return 0
        # d = morph.getFrequency() + constants.getPrior();
        d = self.morph_dict.get(morph, {}).get(state, math.inf) + BaseModel.PRIOR
        #cost = - AmorphousMath.log2(d / (s.classFrequency() + s.classSize()*constants.getPrior()));
        cost =  - math.log2(d / (self.state_freq[state] + self.state_size[state] * BaseModel.PRIOR))
        return cost


    def __get_transition_cost(self, state_a: int, state_b: int) -> float:
        cost = - math.log2((self.transition_freq[state_a][state_b] + BaseModel.PRIOR) / (self.state_freq[state_a] + (self.state_num - 2)* BaseModel.PRIOR))
        return cost

    def __get_simple_lexicon_cost(self, morph: str) -> float:
        return len(morph) + 1 + math.log(len(self.charset) + 2)


    def __get_lexicon_cost(self, state_id: int) -> float:
        #int classSize = lexicon.getMorphList(state).size();
        #double classLexCost = computePrequentialCostForMap(classCount);
        if not state_id or state_id == self.state_num - 1:
            state_lex_cost = self.__get_class_lex_cost([])
        else:
            state_lex_cost = self.__get_class_lex_cost(list(self.state_char_counts[state_id].values()))
        #we add one because Elias coding works only for positive integers but we deal with nonnegative integers.
        #double costOfCodingSize = amorphous.math.AmorphousMath.computeEliasOmegaLength(classSize + 1); 
        cost_of_coding_size = self.__compute_elias_omega_length(self.state_size[state_id] + 1)
        return cost_of_coding_size + state_lex_cost


    def __get_class_lex_cost(self, counts: list) -> float:
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
        i = 1
        lambda_i_number = math.floor(math.log2(x))
        while i >= 1 and lambda_i_number > 0:
            length += lambda_i_number + 1
            i += 1
            lambda_i_number = math.floor(math.log2(lambda_i_number))
        return length

