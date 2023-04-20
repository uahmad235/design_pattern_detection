import os
import shutil

def extract_java_files(src_dir, dest_dir):
    """
    Extracts all Java files from subdirectories of src_dir and moves them to dest_dir.
    """
    java_file_counter = 0  # Counter to keep track of number of Java files extracted

    for dirpath, dirnames, filenames in os.walk(src_dir):
        for filename in filenames:
            if filename.endswith(".java"):
                # Join the directory path and filename to get the full file path
                src_path = os.path.join(dirpath, filename)
                # Create the destination directory if it doesn't exist
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                # Move the Java file to the destination directory
                shutil.move(src_path, os.path.join(dest_dir, filename))
                java_file_counter += 1  # Increment the counter for each Java file extracted

    return java_file_counter

# Set the source directory to the current directory
src_dir = "."  # Current directory
dest_dir = "java"  # Destination directory

# Call the function to extract Java files and get the counter value
num_java_files = extract_java_files(src_dir, dest_dir)

print("Java files extracted and moved to", dest_dir)
print("Number of Java files extracted:", num_java_files)

