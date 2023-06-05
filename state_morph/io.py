
from .core import BaseModel
try:
    # In Python2 import cPickle for better performance
    import cPickle as pickle
except ImportError:
    import pickle
import re
import os
import h5py
import shutil

class StateMorphIO(object):
    """Class for reading and writing state morphologies."""

    def __init__(self, base_path='./'):
        """Initialize StateMorphIO object."""
        self.base_path = os.path.abspath(base_path)

    
    def load_model_from_text_files(self, num_state: int, num_prefix: int, num_suffix: int, segmented_file: str, **kwargs) -> BaseModel:
        """Read state morphology from file."""
        model_params = {
            'num_state': num_state,
            'num_prefix': num_prefix,
            'num_suffix': num_suffix,
            'lexicon':  {},
            'state_freq': {},
            'transition_freq': [],
        }
        model = BaseModel(model_params, **kwargs)
        self.__load_model_file(model, os.path.join(self.base_path, segmented_file))
        return model

    def write_segmented_file(self, model, segmented_file):
        """Write state morphology to a file."""
        os.makedirs(self.base_path, exist_ok=True)
        with open(os.path.join(self.base_path, segmented_file), 'w', encoding='utf-8') as f:
            word2segment = {}
            for segments, cost in model.segmented_corpus:
                tmp = []
                word = ''
                for morph, state in segments:
                    tmp.append(morph)
                    tmp.append(str(state))
                    word += morph
                word2segment[word] = ' '.join(tmp) + '\t' + '{:.4f}'.format(cost)
            for word in sorted(word2segment.keys()):
                f.write(word2segment[word] + '\n')
        
    def load_model_from_binary_file(self, filename: str, **kwargs) -> BaseModel:
        """Read state morphology from binary file."""
        model_data = pickle.load(open(os.path.join(self.base_path, filename), 'rb'))
        model = BaseModel(model_data['model_param'], **kwargs)
        if len(model_data.get('segmented_corpus') or []):
            model.update_segmented_corpus(model_data['segmented_corpus'], update_model=False)
        return model

    def write_binary_model_file(self, model, filename, no_corpus=False):
        """Write state morphology to a binary file."""
        os.makedirs(self.base_path, exist_ok=True)
        model_data = {
            'model_param': model.get_param_dict(),
            'segmented_corpus': not no_corpus and model.segmented_corpus or None,
        }
        pickle.dump(model_data, open(os.path.join(self.base_path, filename), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def write_partition_file(self, partition_id, partition):
        """Write state morphology to a binary file."""
        with h5py.File(os.path.join(self.base_path, 'tmp', 'partition_{}.h5'.format(partition_id)), 'w') as dest_file:
            dest = dest_file.create_dataset(
                'dataset', (len(partition),), dtype=h5py.special_dtype(vlen=str), chunks=True, compression="gzip")
            dest[:] = partition
            dest_file.flush()
            dest_file.close()
    
    def load_partition_file(self, partition_id):
        with h5py.File(os.path.join(self.base_path, 'tmp', 'partition_{}.h5'.format(partition_id)), 'r') as f:
            partition = [x.decode('utf-8') for x in f['dataset']] 
            f.close()
        return partition
    
    def write_temp_model_params(self, model_param):
        pickle.dump(model_param, open(os.path.join(self.base_path, 'tmp', 'model_param.bin'), 'wb'), 
                    protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_temp_model_params(self):
        return pickle.load(open(os.path.join(self.base_path, 'tmp', 'model_param.bin'), 'rb'))
    
    def create_temporary_directory(self):
        shutil.rmtree(os.path.join(self.base_path, 'tmp'), ignore_errors=True)
        os.makedirs(os.path.join(self.base_path, 'tmp'), exist_ok=True)
    
    def remove_temp_files(self):
        shutil.rmtree(os.path.join(self.base_path, 'tmp'), ignore_errors=True)
    
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
        self.__load_segments(model, raw_segments_str)
            