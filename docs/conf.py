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

autodoc_type_aliases = {
    'giatools.typing.NDArray': 'NDArray',
}
