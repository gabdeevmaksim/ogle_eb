import os
import numpy as np
import pandas as pd
import random
import astropy.units as u
from astroquery.gaia import Gaia
from astropy.table import Table
from astropy.coordinates import SkyCoord
from eclipsing_binary.config import get_paths
from eclipsing_binary.core import EclipsingBinary

def read_data(paths):
    """
    Reads and combines data from eclipsing binary, identification, and 
    (optionally) extinction files using Pandas. 
    Creates EclipsingBinary objects from the combined data.

    Args:
        paths (dict): A dictionary containing the paths to the data files:
                        'ecl_file', 'ident_file', and optionally 'ext_file'.

    Returns:
        list: A list of EclipsingBinary objects.
    """

    # Read eclipsing binary data (_ecl) using fixed-width formatted reader
    ecl_col_specs = [(0, 19), (20, 27), (28, 35), (36, 48), (49, 58), (59, 65), (65, 70)]
    ecl_names = ['object_name', 'I_magnitude', 'V_magnitude', 'period_days', 
                 'epoch_of_minimum', 'main_eclipse_dip', 'second_eclipse_dip']
    ecl_df = pd.read_fwf(paths['ecl_file'], colspecs=ecl_col_specs, names=ecl_names, header=None)

    # Read identification data (_ident)
    ident_col_specs = [(0, 19), (20, 24), (25, 36), (37, 49), (50, 66), (67, 82), (83, 99), (100, 116)]
    ident_names = ['object_name', 'type', 'RA_coord', 'DEC_coord', 'OGLE-IV', 'OGLE-III', 'OGLE-II', 'other_names']
    ident_df = pd.read_fwf(paths['ident_file'], colspecs=ident_col_specs, names=ident_names, header=None)

    # Read extinction data (_ext) if provided
    if 'ext_file' in paths and os.path.isfile(paths['ext_file']):
        ext_col_specs = [(2, 14), (15, 30), (31, 41), (42, 55), (56, 63), (65, 72), (73, 82), (83, 92), (93,102), (103,112), (113,121), (122,130), (131,138), (138,149), (150,161)]
        ext_names = ['RA_deg', 'Dec_deg',  'RA_h', 'Dec_h', 'E(V-I)', '-sigma1', '+sigma2', '(V-I)_RC', '(V-I)_0', 'E(V-I)peak', 'E(V-I)sfd', 'box', 'sep', 'RA_coord', 'DEC_coord']
        ext_df = pd.read_fwf(paths['ext_file'], colspecs=ext_col_specs, names=ext_names, header=3)
        ext_df.drop(ext_df.tail(1).index, inplace=True)
    else:
        ext_df = None

     # Merge dataframes
    merged_df = pd.merge(ident_df, ecl_df, on='object_name', how='left')
    if ext_df is not None:
        merged_df = pd.merge(merged_df, ext_df, on=['RA_coord', 'DEC_coord'], how='left')

    # Remove duplicates by object_name
    merged_df.drop_duplicates(subset='object_name', inplace=True)

    # Create EclipsingBinary objects
    eclipsing_binaries = []
    for index, row in merged_df.iterrows():
        # Convert the row to a dictionary
        row_dict = row.to_dict()

        # Remove unnecessary columns
     
        for col in ['OGLE-IV', 'OGLE-III', 'OGLE-II', 'other_names', 
                    'RA_deg', 'Dec_deg', 'RA_h', 'Dec_h', '(V-I)_RC', '(V-I)_0', 'E(V-I)_peak', 'box', 'sep']:
            row_dict.pop(col, None)

        # Convert V_magnitude to float, handling '-'
        if row_dict['V_magnitude'] == '-':
            row_dict['V_magnitude'] = np.nan
        else:
            row_dict['V_magnitude'] = float(row_dict['V_magnitude'])

        # Rename columns to match EclipsingBinary attributes
        row_dict['obj_type'] = row_dict.pop('type', None)
        row_dict['RA'] = row_dict.pop('RA_coord', None)
        row_dict['DEC'] = row_dict.pop('DEC_coord', None)

        # Create extinction dictionary if ext_df is not None
        if ext_df is not None:
            row_dict['extinction'] = {
                'E(V-I)': row_dict.pop('E(V-I)'),
                '-sigma1': row_dict.pop('-sigma1'),
                '+sigma2': row_dict.pop('+sigma2'),
                'E(V-I)sfd': row_dict.pop('E(V-I)sfd')
            }

        # Create EclipsingBinary object with keyword arguments
        binary = EclipsingBinary(**row_dict)
        eclipsing_binaries.append(binary)

    return eclipsing_binaries

def print_coordinates(binaries, output_filename):
    """
    Prints the coordinates of stars from a list of EclipsingBinary objects to a file.

    Args:
        binaries (list): A list of EclipsingBinary objects.
        output_filename (str): The name of the output file.
    """

    with open(output_filename, 'w') as f:
        for binary in binaries:
            if binary.RA is not None and binary.DEC is not None:
                # Format RA and DEC to HH:MM:SS.SS format
                ra_hms = binary.RA
                dec_dms = binary.DEC

                f.write(f"{ra_hms} {dec_dms}\n")

def gaia_cross_match(binaries, 
                    table_name="my_table", 
                    gaia_table="gaiadr3.gaia_source", 
                    radius=1,
                    save_to_file=False):
    """
    Logs into the Gaia archive, uploads a table of sources created from a list of objects,
    updates column flags, performs a crossmatch with a specified Gaia table, and retrieves 
    the results with a custom query.

    GAIA_USERNAME and GAIA_PASSWORD should be writen in your /.bashrc or /.zshrc file.

    Args:
        binaries (list): List of EclipsingBinary objects.
        table_name (str, optional): Name for the uploaded table. 
                                    Defaults to "my_table".
        gaia_table (str, optional): Name of the Gaia table to 
                                    crossmatch with. Defaults to 
                                    "gaiadr3.gaia_source".
        radius (int, optional): Crossmatch radius in arcseconds. 
                                Defaults to 1.
        save_to_file (bool, optional): Save the resulting table in .ecsv format

    Returns:
        astropy.table.Table: Crossmatch results.
    """

    username = os.environ.get('GAIA_USERNAME')
    password = os.environ.get('GAIA_PASSWORD')
    full_table_name = f"user_{username}.{table_name}"
    xmatch_table_name = "xmatch"

    try:
        Gaia.login(user=username, password=password)

        # Check if table already exists (and delete if it does)
        tables = Gaia.load_tables(only_names=True)
        for table in tables:
            if f"user_mgabdeev.{full_table_name}" == table.get_qualified_name():
                print(f"Table '{full_table_name}' already exists. Deleting...")
                Gaia.delete_user_table(table_name=table_name)
                break

        # Create astropy table from EclipsingBinary objects
        source_table = Table([
            [binary.object_name for binary in binaries],
            [binary.RA.replace(':', ' ') for binary in binaries],
            [binary.DEC.replace(':', ' ') for binary in binaries]
        ], names=['object_name', 'ra', 'dec'])

        # Convert 'ra' and 'dec' columns to SkyCoord objects
        coords = SkyCoord(source_table['ra'], source_table['dec'], unit=(u.hourangle, u.deg))

        # Replace 'ra' and 'dec' columns with numeric values
        source_table['ra'] = coords.ra.deg
        source_table['dec'] = coords.dec.deg

        # Upload table to Gaia archive
        try:
            Gaia.upload_table(upload_resource=source_table, 
                              table_name=table_name,
                              format="votable")
        except Exception as e:
            print(f"Error uploading table: {e}")
            return None

        # Update column flags (including 'PK' for 'object_name')
        try:
            Gaia.update_user_table(table_name=full_table_name,
                                  list_of_changes=[["ra","flags","Ra"], 
                                                   ["dec","flags","Dec"],
                                                   ["object_name","flags","Pk"]])
        except Exception as e:
            print(f"Error updating table flags: {e}")
            return None
        
        # Perform crossmatch
        try:
            job = Gaia.cross_match(
                full_qualified_table_name_a=full_table_name,
                full_qualified_table_name_b=gaia_table,
                results_table_name=xmatch_table_name,
                radius=radius,
                background=True,
                verbose=True
            )
        except Exception as e:
            print(f"Error performing crossmatch: {e}")
            return None

        # Retrieve crossmatch results with custom query
        try:
            query = (f"SELECT c.separation*3600 AS separation_arcsec, a.*, b.* "
                     f"FROM gaiadr3.gaia_source_lite AS a, {full_table_name} AS b, "
                     f"user_{username}.{xmatch_table_name} AS c "
                     f"WHERE c.gaia_source_source_id = a.source_id AND "
                     f"c.{table_name}_{table_name}_oid = b.{table_name}_oid")
            job = Gaia.launch_job(query=query)
            results = job.get_results()

            # Write results to file
            if save_to_file:
                paths=get_paths()
                output_filename = f"{paths['demo_data_dir']}xmatch_{table_name}_{gaia_table.replace('.', '_')}.ecsv"
                results.write(output_filename, overwrite=True) 

            return results
        except Exception as e:
            print(f"Error retrieving or saving results: {e}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

    finally:
        # Always logout, even if there's an error
        Gaia.logout()


def select_random_objects(output_file, num_samples=10000):
    """
    Selects 10,000 random 'C' (contact) and 10,000 random 'NC' (non-contact) objects
    from the identification file and writes them to a new file.

    Args:
        output_file (str): The path to the output file where the selected objects will be written.
        num_samples (int): The number of random samples to select for each type ('C' and 'NC'). Default is 10,000.

    Returns:
        None
    """

    # Load the necessary paths
    paths = get_paths()
    ident_file = paths['ident_file']

    # Read identification data (_ident)
    ident_col_specs = [(0, 19), (20, 24), (25, 36), (37, 49), (50, 66), (67, 82), (83, 99), (100, 116)]
    ident_names = ['object_name', 'type', 'RA_coord', 'DEC_coord', 'OGLE-IV', 'OGLE-III', 'OGLE-II', 'other_names']
    ident_df = pd.read_fwf(ident_file, colspecs=ident_col_specs, names=ident_names, header=None)

    # Filter by type 'C' and 'NC'
    contact_binaries = ident_df[ident_df['type'] == 'C']
    non_contact_binaries = ident_df[ident_df['type'] == 'NC']

    # Select random samples
    # Take out 1000 more objects
    contact_samples = contact_binaries.sample(n=num_samples + num_samples//10, replace=True)
    non_contact_samples = non_contact_binaries.sample(n=num_samples + num_samples//10, replace=True)

    lc_I_dir = paths['lc_I_dir']

    # Check the existence of LCs files and filter the samples
    contact_samples['lc_exists'] = contact_samples['object_name'].apply(lambda x: os.path.isfile(os.path.join(lc_I_dir, f"{x}.dat")))
    non_contact_samples['lc_exists'] = non_contact_samples['object_name'].apply(lambda x: os.path.isfile(os.path.join(lc_I_dir, f"{x}.dat")))

    # Count how many LCs were not found
    contact_not_found = contact_samples[~contact_samples['lc_exists']].shape[0]
    non_contact_not_found = non_contact_samples[~non_contact_samples['lc_exists']].shape[0]
    print(f"Contact LCs not found: {contact_not_found}")
    print(f"Non-contact LCs not found: {non_contact_not_found}")

    # Filter the samples to only include those with existing LCs
    existing_contact_samples = contact_samples[contact_samples['lc_exists']]
    existing_non_contact_samples = non_contact_samples[non_contact_samples['lc_exists']]

    # Select the first num_samples objects
    final_contact_samples = existing_contact_samples.head(num_samples)
    final_non_contact_samples = existing_non_contact_samples.head(num_samples)

    # Combine samples
    selected_samples = pd.concat([final_contact_samples, final_non_contact_samples])

    # Write the selected objects to a new file in fixed-width format
    with open(output_file, 'w') as f:
        for index, row in selected_samples.iterrows():
            f.write(f"{row['object_name']:19} {row['type']:4} {row['RA_coord']:12} {row['DEC_coord']:12} {row['OGLE-IV']:17} {row['OGLE-III']:15} {row['OGLE-II']:16} {row['other_names']:16}\n")

