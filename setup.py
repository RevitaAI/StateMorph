from codecs import open
from ez_setup import use_setuptools
use_setuptools()
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension
from datetime import datetime

import re
main_py = open('state_morph/__init__.py', encoding='utf-8').read()
metadata = dict(re.findall("__([a-z]+)__ = '([^']+)'", main_py))

requires = [
    'dask>=2022.1.0',
    'distributed>=2022.1.0',
    'bokeh >= 2.1.1',
    'numpy'
]

ext_modules=[
    Extension("state_morph.core", ["state_morph/core.pyx"]),
]

setup(
    name='StateMorph',
    version=metadata['version'] + datetime.now().strftime('+beta%Y%m%d'),
    author=metadata['author'],
    author_email='revita@cs.helsinki.fi',
    #   url='',
    description='A tool for unsupervised and semi-supervised morphological segmentation',
    packages=['state_morph'],
    package_data={'state_morph': ["py.typed", "*.pyi", "**/*.pyi"]},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.7',
    license="BSD",
    scripts=[],
    install_requires=requires,
    extras_require={},
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3", 'embedsignature': True}),
)