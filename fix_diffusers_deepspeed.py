"""
Compatibility patch for diffusers and transformers deepspeed attribute issue.

This module patches the transformers.deepspeed module to avoid AttributeError
when diffusers tries to check for deepspeed zero3 configuration.
"""

import sys
import importlib.util


def apply_patch():
    """Apply the deepspeed compatibility patch."""
    # Check if transformers is available
    if importlib.util.find_spec("transformers") is None:
        return
    
    import transformers
    
    # Create a mock deepspeed module if it doesn't exist
    if not hasattr(transformers, 'deepspeed'):
        class MockDeepSpeed:
            @staticmethod
            def is_deepspeed_zero3_enabled():
                return False
        
        transformers.deepspeed = MockDeepSpeed()


# Apply patch when this module is imported
apply_patch()
