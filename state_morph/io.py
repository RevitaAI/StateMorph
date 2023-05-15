
from .core import BaseModel

class StateMorphIO(object):
    """Class for reading and writing state morphologies."""

    def __init__(self):
        """Initialize StateMorphIO object."""
        pass

    
    def load_model_from_text_files(self, segmented_file: str, lexicon_file: str, **kwargs) -> None:
        """Read state morphology from file."""
        model_params = {
            'morph_dict':  {},
            'state_freq': {},
            'state_size': {},
            'state_char_counts': {},
            'transition_freq': None,
            'state_num': 0,
            'charset': [],
            'lexicon_costs': [],
            'transition_costs': []
        }
        model = BaseModel(model_params, **kwargs)
        self.__load_model_file(model, segmented_file, lexicon_file)
        return model
        

    def write_binary_file(self, filename):
        """Write state morphology to a binary file."""
        raise NotImplementedError
    
    def __load_lexicon(self, model, raw_lexicon_str: str):
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
            if morph not in model.morph_dict:
                model.morph_dict[morph] = {}
            model.morph_dict[morph][state_id] = float(freq)
            model.state_size[state_id] = model.state_size.get(state_id, 0) + 1
            if state_id not in self.state_char_counts:
                model.state_char_counts[state_id] = {}
            for c in morph:
                model.state_char_counts[state_id][c] = model.state_char_counts[state_id].get(c, 0) + float(freq)
                model.charset.add(c)
        model.state_num = state_id + 1
        model.state_size[0] = 0
        model.state_size[model.state_num - 1] = 0
        model.state_char_counts[0] = {'a': 0}
        model.state_char_counts[model.state_num - 1] = {'a': 0}
    

    def __load_lexicon_file(self, lexicon_file):
        raw_lexicon_str = ''
        with open(lexicon_file, 'r', encoding='utf-8') as f:
            raw_lexicon_str = f.read()
        return raw_lexicon_str


    def __load_segments(self, model, raw_segments_str: str):
        model.transition_freq = [[0 for j in range(model.state_num)] for i in range(model.state_num)]
        for line in raw_segments_str.split('\n'):
            text = line.strip()
            if not len(text):
                continue
            tmp, _ = text.split('\t')
            tmp = tmp.strip().split(' ')
            if len(tmp) > 2:
                word = ''
                for i in range(1, len(tmp) - 2, 2):
                    word += tmp[i-1].strip()
                    state_a = int(tmp[i])
                    state_b = int(tmp[i + 2])
                    model.transition_freq[state_a][state_b] += 1
                    model.state_freq[state_a] = model.state_freq.get(state_a, 0) + 1
            model.state_freq[int(tmp[-1])] = model.state_freq.get(int(tmp[-1]), 0) + 1
            model.state_freq[0] = model.state_freq.get(0, 0) + 1
            model.state_freq[model.state_num - 1] = model.state_freq.get(self.state_num - 1, 0) + 1
            model.transition_freq[0][int(tmp[1])] += 1
            model.transition_freq[int(tmp[-1])][model.state_num - 1] += 1
            if i > 1:
                word += tmp[-2].strip()
            model.corpus.append(word)


    def __load_segmented_file(self, segmented_file) -> str:
        raw_segments_str = ''
        with open(segmented_file, 'r', encoding='utf-8') as f:
            raw_segments_str = f.read()
        return raw_segments_str


    def __load_model_file(self, model, segmented_file: str, lexicon_file: str) -> None:
        raw_lexicon_str = self.__load_lexicon_file(lexicon_file)
        raw_segments_str = self.__load_segmented_file(segmented_file)
        self.__load_raw_model(model, raw_segments_str, raw_lexicon_str)


    def __load_raw_model(self, model, raw_segments_str: str, raw_lexicon_str: str) -> None:
        #print('Read model...')
        self.__load_lexicon(model, raw_lexicon_str)
        self.__load_segments(model, raw_segments_str)
        model.update_costs()
        #print('Num of state:', self.state_num - 2)
        #print('Num of morph:', len(self.morph_dict.keys()))
        