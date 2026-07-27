"""GCG (Greedy Coordinate Gradient) attack methods."""

from .base_gcg import BaseGCG
from .vanilla_gcg import VanillaGCG

__all__ = ["BaseGCG", "VanillaGCG"]
