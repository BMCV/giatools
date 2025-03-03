import sys

if sys.version_info < (3, 11):
    from typing_extensions import *  # noqa: F401, F403
else:
    from typing import *  # noqa: F401, F403
