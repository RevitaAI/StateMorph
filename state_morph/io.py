
from .core import BaseModel
import pickle
import re

class StateMorphIO(object):
    """Class for reading and writing state morphologies."""

    def __init__(self):
        """Initialize StateMorphIO object."""
        pass

    
    def load_model_from_text_files(self, segmented_file: str, **kwargs) -> None:
        """Read state morphology from file."""
        model_params = {
            'morph_dict':  {},
            'state_freq': {},
            'state_size': {},
            'state_char_counts': {},
            'transition_freq': [],
        }
        model = BaseModel(model_params, **kwargs)
        self.__load_model_file(model, segmented_file)
        return model

    def write_segmented_file(self, model, segmented_file):
        """Write state morphology to a file."""
        with open(segmented_file, 'w', encoding='utf-8') as f:
            for segments, cost in model.segmented_corpus:
                tmp = []
                for morph, state in segments:
                    tmp.append(morph)
                    tmp.append(str(state))
                f.write(' '.join(tmp) + '\t' + '{:.4f}'.format(cost) + '\n')
        
    def load_model_from_binary_file(self, filename: str, **kwargs) -> None:
        """Read state morphology from binary file."""
        model_data = pickle.load(open(filename, 'rb'))
        model = BaseModel(model_data['model_param'], **kwargs)
        model.update_segmented_corpus(model_data['segmented_corpus'], update_model=False)
        return model

    def write_binary_file(self, model, filename):
        """Write state morphology to a binary file."""
        model_data = {
            'model_param': model.get_param_dict(),
            'segmented_corpus': model.segmented_corpus,
        }
        pickle.dump(model_data, open(filename, 'wb'))

    
    
    def __load_segments(self, model, raw_segments_str: str):
        segmented_corpus = []
        for line in raw_segments_str.split('\n'):
            text = line.strip()
            if not len(text):
                continue
            tmp, cost = text.split('\t')
            tmp = tmp.strip().split(' ')
            if len(tmp) >= 2:
                segments = []
                for i in range(len(tmp) // 2):
                    morph = tmp[i * 2]
                    state = int(tmp[i * 2 + 1])
                    segments.append((morph, state))
                segmented_corpus.append((segments, float(cost)))
        model.update_segmented_corpus(segmented_corpus)
                


    def __load_segmented_file(self, segmented_file) -> str:
        raw_segments_str = ''
        with open(segmented_file, 'r', encoding='utf-8') as f:
            raw_segments_str = f.read()
        return raw_segments_str


    def __load_model_file(self, model, segmented_file: str) -> None:
        raw_segments_str = self.__load_segmented_file(segmented_file)
        self.__load_raw_model(model, raw_segments_str)


    def __load_raw_model(self, model, raw_segments_str: str) -> None:
        #print('Read model...')
        self.__load_segments(model, raw_segments_str)
        #print('Num of state:', self.state_num - 2)
        #print('Num of morph:', len(self.morph_dict.keys()))
        