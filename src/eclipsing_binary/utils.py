import numpy as np
from src.eclipsing_binary.core import EclipsingBinary

def read_data(filename, ident_filename=None):
    """
    Reads a file with fixed-width columns containing eclipsing binary data,
    optionally combines it with identification data, and returns a list of 
    EclipsingBinary objects.

    Args:
      filename (str): The path to the main data file.
      ident_filename (str, optional): The path to the identification data file.

    Returns:
      list: A list of EclipsingBinary objects.
    """

    column_names = ['object_name', 'I_magnitude', 'V_magnitude', 'period_days', 
                    'epoch_of_minimum', 'main_eclipse_dip', 'second_eclipse_dip']
    
    # Define the widths of each column
    col_widths = [19, 7, 7, 13, 12, 6, 6]  

    eclipsing_binaries = []

    with open(filename, 'r') as file:
        for line in file:
            # Split the line into fields based on column widths
            values = [line[sum(col_widths[:i]):sum(col_widths[:i+1])].strip() 
                      for i in range(len(col_widths))]

            # Convert numeric values to floats, handling '-'
            for i in range(1, len(values)):  # Start from index 1 to skip 'object_name'
                try:
                    values[i] = float(values[i])
                except ValueError:
                    if values[i] == '-':
                        values[i] = np.nan  # Replace '-' with np.nan
                    else:
                        raise  # Raise the error for other conversion issues

            # Create an EclipsingBinary object and add it to the list
            eclipsing_binaries.append(EclipsingBinary(*values))  # Pass values directly
    
        if ident_filename:
          ident_data = {}
          with open(ident_filename, 'r') as ident_file:
              for line in ident_file:
                  parts = line.split()
                  ident_data[parts[0]] = {
                      'obj_type': parts[1],
                      'RA': parts[2],
                      'DEC': parts[3]
                  }

          # Add identification data to EclipsingBinary objects
          for binary in eclipsing_binaries:
              if binary.object_name in ident_data:
                  binary.obj_type = ident_data[binary.object_name]['obj_type']
                  binary.RA = ident_data[binary.object_name]['RA']
                  binary.DEC = ident_data[binary.object_name]['DEC']

    return eclipsing_binaries