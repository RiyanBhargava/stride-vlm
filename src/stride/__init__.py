'''STRIDE: training-free semantic residual routing for frozen VLMs.'''

from .config import RouterConfig
from .routing import METHODS, route
from .types import RouteResult, RoutingContext

__all__ = ['METHODS', 'RouteResult', 'RouterConfig', 'RoutingContext', 'route']
__version__ = '0.2.0'
