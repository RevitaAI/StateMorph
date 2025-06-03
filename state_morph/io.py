
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
from copy import deepcopy

class StateMorphIO(object):
    """Class for reading and writing state morphologies."""

    def __init__(self, base_path='./', charset=None):
        """
        Initialize StateMorphIO object.
        
        Parameters
        ----------
        base_path : str
            Base path for reading and writing files.
        
        """
        self.base_path = os.path.abspath(base_path)
        self.__charset = charset
        if self.__charset is None:
            self.__charset = set()
    
    def set_charset(self, charset: set) -> None:
        """
        Set charset.
        
        Parameters
        ----------
        charset : set
            Charset.
        
        """
        self.__charset = charset
        
    def dump_charset(self, model: BaseModel, filename: str) -> None:
        """
        Dump model current charset to a file.
        
        Parameters
        ----------
        model : BaseModel
            StateMorph model.
        
        """
        charset = set(''.join({morph for morph, state in model.lexicon.keys()}))
        pickle.dump(charset, open(os.path.join(self.base_path, filename), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_model_from_text_files(self, num_state: int, num_prefix: int, num_suffix: int, has_word_boundary: bool,
                                   segmented_file: str, build_cache=False, **kwargs) -> BaseModel:
        """
        Read StateMorph from text file.
        
        Parameters
        ----------
        num_state : int
            Number of states.
        num_prefix : int
            Number of prefix states.
        num_suffix : int
            Number of suffix states.
        has_word_boundary : bool
            If True, the model has word boundary.
        segmented_file : str
            File name of segmented corpus.
        build_cache : bool
            If True, build cache for the model. Default is False.
        **kwargs : dict
        
        """
        from .utils import empty_model_param
        model_params = empty_model_param(
            num_state, num_prefix, num_suffix, kwargs.get('transition_ctrl', {}), has_word_boundary)
        model = BaseModel(model_params, **kwargs)
        self.__load_model_file(model, os.path.join(self.base_path, segmented_file), build_cache)
        return model

    def write_segmented_file(self, model: BaseModel, segmented_file: str) -> None:
        """
        Write StateMorph model to a file.
        
        Parameters
        ----------
        model : BaseModel
            StateMorph model.
        segmented_file : str
            File name of segmented corpus.
        
        
        """
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
        
    def load_model_from_binary_file(self, filename: str, build_cache=False, **kwargs) -> BaseModel:
        """
        Read StateMorph model from binary file.
        
        Parameters
        ----------
        filename : str
            File name of binary file.
        build_cache : bool
            If True, build cache for the model. Default is False.
        
        Returns
        -------
        model : BaseModel
            StateMorph model.
        
        
        """
        model_data = pickle.load(open(os.path.join(self.base_path, filename), 'rb'))
        model = BaseModel(model_data['model_param'], **kwargs)
        if len(model_data.get('segmented_corpus') or []):
            model.update_segmented_corpus(model_data['segmented_corpus'], update_model=False, build_cache=build_cache)
        return model

    def write_binary_model_file(self, model: BaseModel, filename: str, no_corpus=False) -> None:
        """
        Write StateMorph model to a binary file.
        
        Parameters
        ----------
        model : BaseModel
            StateMorph model.
        filename : str
            File name of binary file.
        no_corpus : bool
            If True, do not write segmented corpus to the file. Default is False.
        
        
        """
        os.makedirs(self.base_path, exist_ok=True)
        model_data = {
            'model_param': model.get_param_dict(),
            'segmented_corpus': not no_corpus and model.segmented_corpus or None,
        }
        pickle.dump(model_data, open(os.path.join(self.base_path, filename), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def write_partition_file(self, partition_id: int, partition: list) -> None:
        """
        Write partition to a temporary file.
        
        Parameters
        ----------
        partition_id : int
            Partition id.
        partition : list of str
            List of words in the partition.
        
        
        """
        with h5py.File(os.path.join(self.base_path, 'tmp', 'partition_{}.h5'.format(partition_id)), 'w') as dest_file:
            dest = dest_file.create_dataset(
                'dataset', (len(partition),), dtype=h5py.special_dtype(vlen=str), chunks=True, compression="gzip")
            dest[:] = partition
            dest_file.flush()
            dest_file.close()
    
    def load_partition_file(self, partition_id) -> list:
        '''
        Load partition from a temporary file.
        
        Parameters
        ----------
        partition_id : int
            Partition id.
        
        Returns
        -------
        partition : list of str
            List of words in the partition.
        
        '''
        with h5py.File(os.path.join(self.base_path, 'tmp', 'partition_{}.h5'.format(partition_id)), 'r') as f:
            partition = [x.decode('utf-8') for x in f['dataset']] 
            f.close()
        return partition
    
    def write_temp_file(self, filename: str, data: object) -> None:
        '''
        Write data to a temporary file.
        
        Parameters
        ----------
        filename : str
            File name.
        data : object
            Data to be written to the file.
        
        '''
        pickle.dump(data, open(os.path.join(self.base_path, 'tmp', filename), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
    def load_temp_file(self, filename: str) -> object:
        '''
        Load data from a temporary file.
        
        Parameters
        ----------
        filename : str
            File name.
        
        Returns
        -------
        data : object
            Data loaded from the file.
        
        '''
        try:
            return pickle.load(open(os.path.join(self.base_path, 'tmp', filename), 'rb'))
        except:
            return None
    
    def write_temp_model_params(self, model_param: dict) -> None:
        '''
        Write model parameters to a temporary file.
        
        Parameters
        ----------
        model_param : dict
            Model parameters.
        
        '''
        params = deepcopy(model_param)
        if self.__charset:
            params['charset'] = self.__charset
        pickle.dump(params, open(os.path.join(self.base_path, 'tmp', 'model_param.bin'), 'wb'), 
                    protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_temp_model_params(self) -> dict:
        '''
        Load model parameters from a temporary file.
        
        Returns
        -------
        model_param : dict
            Model parameters.
        
        
        '''
        return pickle.load(open(os.path.join(self.base_path, 'tmp', 'model_param.bin'), 'rb'))
    
    def create_temporary_directory(self) -> None:
        '''
        Create a temporary directory for storing temporary files.
        '''
        
        
        shutil.rmtree(os.path.join(self.base_path, 'tmp'), ignore_errors=True)
        os.makedirs(os.path.join(self.base_path, 'tmp'), exist_ok=True)
    
    def remove_temp_files(self) -> None:
        '''
        Remove temporary files.
        '''
        shutil.rmtree(os.path.join(self.base_path, 'tmp'), ignore_errors=True)
    
    def __load_segments(self, model, raw_segments_str: str, build_cache: bool) -> None:
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
        model.update_segmented_corpus(segmented_corpus, build_cache=build_cache)
                


    def __load_segmented_file(self, segmented_file) -> str:
        raw_segments_str = ''
        with open(segmented_file, 'r', encoding='utf-8') as f:
            raw_segments_str = f.read()
        return raw_segments_str


    def __load_model_file(self, model, segmented_file: str, build_cache: bool) -> None:
        raw_segments_str = self.__load_segmented_file(segmented_file)
        self.__load_segments(model, raw_segments_str, build_cache)