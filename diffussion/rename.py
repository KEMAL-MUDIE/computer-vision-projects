import os
# Specify the directory where your images are located
directory = "class1(161+)_samples"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Split the filename into base and extension
    base, extension = os.path.splitext(filename)
    # Check if the filename ends with '_2'
    if base.endswith('_2'):
        # Generate the new filename
        new_base = base[:-1] + '1'  # Remove the last character '2' and add '1'
        new_filename = new_base + extension
        # Generate the full file paths
        current_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(current_path, new_path)

print(f"Renamed done")