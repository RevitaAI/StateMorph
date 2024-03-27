__all__ = ['StateMorphTrainer', 'StateMorphIO', 'BaseModel', 'get_version']

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("StateMorph")
except PackageNotFoundError:
    # package is not installed
    __version__ = 'Unknown'
__author__ = 'Jue Hou'
__author_email__ = "revita@cs.helsinki.fi"

def get_version():
    return version

from .io import StateMorphIO
from .core import BaseModel
from .trainer import StateMorphTrainer