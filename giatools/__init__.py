from .image import Image  # noqa: F401

VERSION_MAJOR = 0
VERSION_MINOR = 3
VERSION_PATCH = 1

VERSION = '%d.%d%s' % (VERSION_MAJOR, VERSION_MINOR, '.%d' % VERSION_PATCH if VERSION_PATCH > 0 else '')
