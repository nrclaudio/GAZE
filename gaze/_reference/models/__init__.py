"""
Core model architectures for metabolite prediction
"""

# Import from the original losses.py file, not the losses/ directory
try:
    from .losses import MaskedMetaboliteLoss
    __all__ = ['MaskedMetaboliteLoss']
except ImportError:
    # If there are import issues, just skip
    __all__ = []
