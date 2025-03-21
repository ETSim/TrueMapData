# filepath: /c/Users/antoi/OneDrive/Bureau/tmd/examples/demo_polyscope.py
'''
This is a symbolic link to the main demo_polyscope.py file.
Please run the demo using:
    python demo_polyscope.py
'''

import os
import sys

# Execute the main demo file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
exec(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo_polyscope.py')).read())
