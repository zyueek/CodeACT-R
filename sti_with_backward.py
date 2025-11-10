import pandas as pd
import re
import argparse
import random
import os
import warnings
import statistics
from tqdm import *
def find_variable_and_statements_from_file(lines):
    try:
        # Regex to find 'return <variable>'
        variable_match = None
        for i, line in enumerate(lines):
            match = re.search(r'return (\w+)', line)
            if match:
                variable_match = (match.group(1), i + 1)     
        if variable_match:
            variable, return_line = variable_match
            # Find all statements including the variable
            statements = []
            for i, line in enumerate(lines):
                if variable in line:
                    statements.append((line.strip(), i + 1))
            return variable, return_line, statements
        else:
            return None, None, []
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
        return None, None, []
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None, []
def set_backward(filename,number,seed):
  variable, return_line, statements = find_variable_and_statements_from_file(filename)


