"""
Pytest fixtures shared across test modules.
"""
import os
import tempfile
import struct
import numpy as np
import pytest

@pytest.fixture
def struct_module():
    """
    Return the struct module for use in tests.
    This is needed because some test files might not have imported struct.
    """
    import struct
    return struct
