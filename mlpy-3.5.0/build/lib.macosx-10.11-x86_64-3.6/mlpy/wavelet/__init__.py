"""Wavelet transform
"""

import sys
if sys.version >= '3':
    from ._dwt import *
    from ._uwt import *
else:
    from _dwt import *
    from _uwt import *

from .continuous import *
from . import continuous
from .uwt_align import *
from . import uwt_align
from .padding import *
from . import padding


__all__ = []
__all__ += continuous.__all__
__all__ += uwt_align.__all__
__all__ += ["dwt", "idwt", "uwt", "iuwt"]
__all__ += padding.__all__
