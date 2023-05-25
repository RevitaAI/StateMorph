__all__ = ['StateMorphTrainer', 'StateMorphIO', 'BaseModel']

__version__ = '0.0.1'
__author__ = 'Jue Hou'
__author_email__ = "revita@cs.helsinki.fi"

def get_version():
    return __version__

from .io import StateMorphIO
from .core import BaseModel
from .trainer import StateMorphTrainer