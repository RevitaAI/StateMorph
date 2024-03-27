__all__ = ['StateMorphTrainer', 'StateMorphIO', 'BaseModel']
from ._version import version, __version__

__author__ = 'Jue Hou'
__author_email__ = "revita@cs.helsinki.fi"

def get_version():
    return version

from .io import StateMorphIO
from .core import BaseModel
from .trainer import StateMorphTrainer