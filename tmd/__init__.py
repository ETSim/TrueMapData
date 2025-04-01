"""
TMD (True Map Data) Package.

A package for working with topographic surface data, including reading,
writing, visualizing, and analyzing TMD files.
"""

__version__ = "2.0.0"

# Import the main exception classes for easy access
from tmd.exceptions import TMDException, TMDFileError, TMDVersionError, TMDDataError

# Import core functionality - use updated imports
from tmd.core.tmd import TMDProcessingError
from tmd.core import TMD, TMDProcessor, load, get_registered_plotters
from tmd.utils.files import TMDFileUtilities
import tmd.plotters