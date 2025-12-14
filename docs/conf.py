project = 'giatools'
copyright = '2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University'
author = 'Leonid Kostrykin'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_autorun',
]

python_display_short_literal_types = True
autodoc_typehints = 'signature'
autodoc_typehints_format = 'short'
autodoc_type_aliases = {
    'NDArray': 'giatools.NDArray',
    'giatools.NDArray': 'giatools.NDArray',
    'giatools.typing.NDArray': 'giatools.NDArray',
}
