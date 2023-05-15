import math
import json
import os
import sys
import re
from copy import deepcopy

class Segmenter(object):
    PRIOR = 0.5
    LOG_2 = math.log(2)
    HALF_LOG_2_PI = 0.5 * math.log(2.0 * math.pi)
    MORPH_SIZE = 8
    def __init__(self, dummy_segment_states, **kwargs):
        '''
        # Load by using segmentation and lexicon file, which is the direct output from java code
        # A model param json file will be dumped
        # segmenter = Segmenter(segmented_file='segmented_file_path', lexicon_file='lexicon_path', model_path='./')

        # Load by using direct string from segmentation and lexicon file
        # A model param json file will be dumped
        # Suitable for multiprocessing
        # raw_segments_str = ''
        # raw_lexicon_str = ''
        # with open(lexicon_path, 'r', encoding='utf-8') as f:
        #     raw_lexicon_str = f.read()
        # with open(model_path, 'r', encoding='utf-8') as f:
        #     raw_segments_str = f.read()
        # segmenter = Segmenter(raw_segments_str=raw_segments_str, raw_lexicon_str=raw_lexicon_str, model_path='./')

        # Load by using model param dict, which is read from the json file (recommended)
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
        segmenter.debug_segment('aaltoenergiaa', [('aalto', 4), ('energia', 11), ('a', 8)], 31.6421)
        '''
        self.dummy_segment_states = dummy_segment_states   
        self.morph_dict = {}
        self.state_freq = {}
        self.state_size = {}
        self.state_counts = {}
        self.transition_freq = None
        self.state_num = 0
        self.charset = set()
        self.lexicon_costs = []
        self.transition_costs = []
        if 'segmented_file' in kwargs and 'lexicon_file' in kwargs:
            dump_path = kwargs.get('model_path', os.path.join( os.path.realpath(sys.path[0]), 'StateMorph_model_param.json'))
            self.__load_model_file(kwargs['segmented_file'], kwargs['lexicon_file'], dump_path)
        elif 'raw_segments_str' in kwargs and 'raw_lexicon_str' in kwargs:
            dump_path = kwargs.get('model_path', os.path.join( os.path.realpath(sys.path[0]), 'StateMorph_model_param.json'))
            self.__load_raw_model(kwargs['raw_segments_str'], kwargs['raw_lexicon_str'], dump_path)
        elif 'model_params' in kwargs:
            self.__load_model_params(kwargs['model_params'])
        else:
            raise Exception('No model loaded')

    
    def save_model_params(self, path):
        model_params = {
            'morph_dict': self.morph_dict,
            'state_freq': self.state_freq,
            'state_size': self.state_size,
            'state_counts': self.state_counts,
            'transition_freq': self.transition_freq,
            'state_num': self.state_num,
            'charset': list(self.charset),
            'lexicon_costs': self.lexicon_costs,
            'transition_costs': self.transition_costs
        }
        json.dump(model_params, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
    
    def __load_model_params(self, model_params:dict):
        self.morph_dict = {k: {int(vk): vv for vk, vv in v.items()} for k, v in model_params['morph_dict'].items()}
        self.state_freq = {int(k): v for k, v in model_params['state_freq'].items()}
        self.state_size = {int(k): v for k, v in model_params['state_size'].items()}
        self.state_counts = {int(k): v for k, v in model_params['state_counts'].items()}
        self.transition_freq = model_params['transition_freq']
        self.state_num = model_params['state_num']
        self.charset = set(model_params['charset'])
        self.lexicon_costs = model_params['lexicon_costs']
        self.transition_costs = model_params['transition_costs']

        
    def __load_lexicon(self, raw_lexicon_str: str):
        state_id = 0
        for line in raw_lexicon_str.split('\n'):
            text = line.strip()
            if not len(text):
                continue
            if text.startswith('Class'):
                _, state_id = text.split(' ')
                state_id = int(state_id)
                continue
            morph, _, freq = text.split(' ')
            if morph not in self.morph_dict:
                self.morph_dict[morph] = {}
            self.morph_dict[morph][state_id] = float(freq)
            self.state_size[state_id] = self.state_size.get(state_id, 0) + 1
            if state_id not in self.state_counts:
                self.state_counts[state_id] = {}
            for c in morph:
                self.state_counts[state_id][c] = self.state_counts[state_id].get(c, 0) + float(freq)
                self.charset.add(c)
        self.state_num = state_id + 1
        self.state_size[0] = 0
        self.state_size[self.state_num - 1] = 0
        self.state_counts[0] = {'a': 0}
        self.state_counts[self.state_num - 1] = {'a': 0}
    

    def __load_lexicon_file(self, lexicon_file):
        raw_lexicon_str = ''
        with open(lexicon_file, 'r', encoding='utf-8') as f:
            raw_lexicon_str = f.read()
        return raw_lexicon_str


    def __load_segments(self, raw_segments_str: str):
        self.transition_freq = [[0 for j in range(self.state_num)] for i in range(self.state_num)]
        for line in raw_segments_str.split('\n'):
            text = line.strip()
            if not len(text):
                continue
            tmp, _ = text.split('\t')
            tmp = tmp.strip().split(' ')
            if len(tmp) > 2:
                for i in range(1, len(tmp) - 2, 2):
                    state_a = int(tmp[i])
                    state_b = int(tmp[i + 2])
                    self.transition_freq[state_a][state_b] += 1
                    self.state_freq[state_a] = self.state_freq.get(state_a, 0) + 1
            self.state_freq[int(tmp[-1])] = self.state_freq.get(int(tmp[-1]), 0) + 1
            self.state_freq[0] = self.state_freq.get(0, 0) + 1
            self.state_freq[self.state_num - 1] = self.state_freq.get(self.state_num - 1, 0) + 1
            self.transition_freq[0][int(tmp[1])] += 1
            self.transition_freq[int(tmp[-1])][self.state_num - 1] += 1


    def __load_segmented_file(self, segmented_file) -> str:
        raw_segments_str = ''
        with open(segmented_file, 'r', encoding='utf-8') as f:
            raw_segments_str = f.read()
        return raw_segments_str


    def __load_model_file(self, segmented_file: str, lexicon_file: str, dump_path: str) -> None:
        raw_lexicon_str = self.__load_lexicon_file(lexicon_file)
        raw_segments_str = self.__load_segmented_file(segmented_file)
        self.__load_raw_model(raw_segments_str, raw_lexicon_str, dump_path)


    def __load_raw_model(self, raw_segments_str: str, raw_lexicon_str: str, dump_path: str) -> None:
        #print('Read model...')
        self.__load_lexicon(raw_lexicon_str)
        self.__load_segments(raw_segments_str)
        self.lexicon_costs = [self.__get_lexicon_cost(_) for _ in range(1, self.state_num)]
        self.transition_costs = [[self.__get_transition_cost(i, j) for j in range(self.state_num)] for i in range(self.state_num)]
        #print('Num of state:', self.state_num - 2)
        #print('Num of morph:', len(self.morph_dict.keys()))
        if dump_path:
            self.save_model_params(dump_path)


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

    def __search(self, word: str) -> tuple :
        dp_matrix = [[{'state': state, 'char_index': char_index, 'cost': math.inf, 'previous': None, 'morph': ''} 
                      for state in range(self.state_num)] 
                    for char_index in range(len(word) + 2)]
        dp_matrix[0][0]['cost'] = 0
        for row in range(1, len(word) + 2):
            for col in range(1, self.state_num):
                if (row != len(word) + 1 and col != self.state_num - 1) or \
                   (row == len(word) + 1 and col == self.state_num - 1):
                    current_cell = dp_matrix[row][col]
                    search_space = [cell for rows in dp_matrix[max(row - Segmenter.MORPH_SIZE, 0): row] for cell in rows[: -1]]
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

    def segment(self, words: str, cache={}) -> dict:
        re_pattern = re.compile('([' + ''.join(self.charset) + ''.join(self.charset).upper() +']+)|(\d+)|(\W)|(\s+)')
        raw_segment = [x for x in re.split(re_pattern, words) if x and len(x.strip())]
        segments = []
        cost = 0
        for subword in raw_segment:
            if set(subword.lower()).issubset(self.charset):
                seg, c = cache.get(subword.lower(), self.__search(subword.lower()))
                if subword.lower() not in cache:
                    cache[subword.lower()] = (seg, c)
                seg = deepcopy(seg)
                if subword.istitle():
                    seg[0] = (seg[0][0].title(), seg[0][1])
                elif subword.isupper():
                    seg = [(s[0].upper(), s[1]) for s in seg]
                if self.dummy_segment_states:
                    seg = [(_[0], 0) for _ in seg]
                segments.extend(seg)
                cost += c
            else:
                if subword.isnumeric():
                    state = -1
                elif subword.isalpha():
                    state = -2
                else:
                    state = -3
                segments.append((subword, state))
                cost += 0
            segments.append(('[SEP]', -5))
        segments = segments[:-1]
        return {
            'morphs': list(map(lambda x: x[0], segments)),
            'states': list(map(lambda x: x[1], segments)),
            'segment': segments,
            'cost': cost
        }

    def debug_segment(self, word:str, expected_segment: list, expected_cost: float) -> None:
        segment, cost = self.__search(word)
        print('If same segment', len(segment)== len(expected_segment) and all(i==j for i, j in zip(segment, expected_segment)))
        print('Cost prec error:', float(format((cost - expected_cost) / expected_cost, '.5f')), '%')


    def __get_emission_cost(self, morph: str, state: int) -> float:
        if morph == '' and state == self.state_num - 1:
            return 0
        # d = morph.getFrequency() + constants.getPrior();
        d = self.morph_dict.get(morph, {}).get(state, math.inf) + Segmenter.PRIOR
        #cost = - AmorphousMath.log2(d / (s.classFrequency() + s.classSize()*constants.getPrior()));
        cost =  - math.log2(d / (self.state_freq[state] + self.state_size[state] * Segmenter.PRIOR))
        return cost


    def __get_transition_cost(self, state_a: int, state_b: int) -> float:
        cost = - math.log2((self.transition_freq[state_a][state_b] + Segmenter.PRIOR) / (self.state_freq[state_a] + (self.state_num - 2)* Segmenter.PRIOR))
        return cost

    def __get_simple_lexicon_cost(self, morph: str) -> float:
        return len(morph) + 1 + math.log(len(self.charset) + 2)


    def __get_lexicon_cost(self, state_id: int) -> float:
        #int classSize = lexicon.getMorphList(state).size();
        #double classLexCost = computePrequentialCostForMap(classCount);
        if not state_id or state_id == self.state_num - 1:
            state_lex_cost = self.__get_class_lex_cost([])
        else:
            state_lex_cost = self.__get_class_lex_cost(list(self.state_counts[state_id].values()))
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
            cost -= math.lgamma(d + Segmenter.PRIOR) / Segmenter.LOG_2
            cost += math.lgamma(Segmenter.PRIOR) / Segmenter.LOG_2
            sumOfEvents += d + Segmenter.PRIOR
            sumOfPriors += Segmenter.PRIOR

        cost += math.lgamma(sumOfEvents) / Segmenter.LOG_2
        cost -= math.lgamma(sumOfPriors) / Segmenter.LOG_2
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

