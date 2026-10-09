__version__ = "2.3.0"
from .spectralData import SpectralData
from .freq_domain import *
from .time_domain import *
from . import tools
from .multiaxial import *

_LAZY = {
    'visualize': ('.visualize', None),
    'set_mesh': ('.visualize', 'set_mesh'),
    'pick_point': ('.visualize', 'pick_point'),
    'PSDgen': ('.gui', 'PSDgen'),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        mod = importlib.import_module(module, __name__)
        return mod if attr is None else getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [n for n in dir() if not n.startswith('_')] + ['set_mesh', 'pick_point']
