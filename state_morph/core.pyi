from _typeshed import Incomplete

class BaseModel:
    PRIOR: float
    LOG_2: Incomplete
    HALF_LOG_2_PI: Incomplete
    MORPH_SIZE: int
    num_state: int
    lexicon: Incomplete
    state_freq: Incomplete
    num_prefix: int
    num_suffix: int
    transition_freq: Incomplete
    lexicon_costs: Incomplete
    transition_costs: Incomplete
    segmented_corpus: Incomplete
    def __init__(self, model_param) -> None: ...
    def get_param_dict(self): ...
    def update_counts(self): ...
    def update_segmented_corpus(self, segmented_corpus, update_model: bool = ...) -> None: ...
    def train_step(self, corpus=..., temperature: int = ..., is_final: bool = ...): ...
    def update_model(self) -> None: ...
    def compute_encoding_cost(self) -> float: ...
    def segment(self, word: str) -> tuple: ...
    def debug_dp_matrix(self, word, dp_matrix, segment) -> None: ...
    def debug_segment(self, word: str, expected_segment: list, expected_cost: float) -> None: ...