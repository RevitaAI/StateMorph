from .core import BaseModel

def _map_step(partition_id, model_param, segmented_corpus):
    """Map step function for multiprocessing."""
    print('Map ID:', partition_id, 'Corpus size:', len(segmented_corpus), 'started...')
    model = BaseModel(model_param)
    model.update_segmented_corpus(segmented_corpus, update_model=False)
    model_param, segmented_corpus = model.train_step()
    print('Map ID:', partition_id, 'ended...')
    return model_param, segmented_corpus

def _reduce_step(total_model_param, total_corpus, model_param, segmented_corpus):
    """Reduce step function for multiprocessing."""
    
    total_corpus += segmented_corpus
    
    for k, v in model_param['morph_dict'].items():
        if k not in total_model_param['morph_dict']:
            total_model_param['morph_dict'][k] = {}
        for vk, vv in v.items():            
            if vk not in total_model_param['morph_dict'][k]:
                total_model_param['morph_dict'][k][vk] = 0
            total_model_param['morph_dict'][k][vk] += vv
    for k, v in model_param['state_freq'].items():
        if k not in total_model_param['state_freq']:
            total_model_param['state_freq'][k] = 0
        total_model_param['state_freq'][k] += v
    for k, v in model_param['state_size'].items():
        if k not in total_model_param['state_size']:
            total_model_param['state_size'][k] = 0
        total_model_param['state_size'][k] += v
    for k, v in model_param['state_char_counts'].items():
        if k not in total_model_param['state_char_counts']:
            total_model_param['state_char_counts'][k] = {}
        for vk, vv in v.items():
            if vk not in total_model_param['state_char_counts'][k]:
                total_model_param['state_char_counts'][k][vk] = 0
            total_model_param['state_char_counts'][k][vk] += vv
            
    if not len(total_model_param['transition_freq']):
        total_model_param['transition_freq'] = model_param['transition_freq']
    else:
        for i in range(len(model_param['transition_freq'])):
            for j in range(len(model_param['transition_freq'][i])):
                total_model_param['transition_freq'][i][j] += model_param['transition_freq'][i][j]