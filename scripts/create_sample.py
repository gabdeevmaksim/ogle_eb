import os
import sys

sys.path.append(os.path.abspath('./src/')) 

from eclipsing_binary.utils import select_random_objects

def main():
    output_file = 'demo_data/blg_phot/selected_ident.dat'
    num_samples = 10000

    # Create a sample of 10,000 random 'C' and 10,000 random 'NC' objects
    select_random_objects(output_file, num_samples)

    print(f"Sample created and written to {output_file}")

if __name__ == "__main__":
    main()