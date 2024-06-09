import os
import shutil

def duplicate_images(original_path, destination_path, num_copies):
    # Check if the destination directory exists, create if not
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Counter for the number of copied images
    num_copied_images = 0

    # Iterate over each image in the original directory
    for root, dirs, files in os.walk(original_path):
        for file in files:
            # Construct the full path of the original image
            original_image_path = os.path.join(root, file)

            # Get the file extension
            file_extension = os.path.splitext(file)[1]

            # Generate new file names with c1, c2, c3, c4 prefixes
            new_file_names = [f'c{i + 1}_{file}' for i in range(num_copies)]

            # Duplicate the image with the new names to the destination directory
            for new_file_name in new_file_names:
                new_image_path = os.path.join(destination_path, new_file_name)
                shutil.copyfile(original_image_path, new_image_path)
                num_copied_images += 1

    print(f"Number of images in the destination directory: {num_copied_images}")

if __name__ == "__main__":
    original_path = 'class_4'  # Replace with your actual path
    destination_path = 'class_4n'  # Replace with your desired destination path
    num_copies = 4

    duplicate_images(original_path, destination_path, num_copies)
