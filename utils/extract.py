import os
import shutil

def process_repo_folder(repo_dir):
    """
    Processes a repo folder by moving all Java files to a temp_java folder and removing everything else.
    """
    java_file_counter = 0  # Counter to keep track of the number of Java files moved

    # Create a temp_java folder if it doesn't exist
    temp_java_dir = os.path.join(repo_dir, 'temp_java')
    if not os.path.exists(temp_java_dir):
        os.makedirs(temp_java_dir)

    for dirpath, dirnames, filenames in os.walk(repo_dir):
        for filename in filenames:
            if filename.endswith(".java"):
                # Join the directory path and filename to get the full file path
                src_path = os.path.join(dirpath, filename)
                # Move the Java file to the temp_java folder
                dst_path = os.path.join(temp_java_dir, filename)
                if not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
                    java_file_counter += 1  # Increment the counter for each Java file moved

    # Remove everything else (files and folders) from the repo folder except for the temp_java folder
    for item in os.listdir(repo_dir):
        item_path = os.path.join(repo_dir, item)
        if item != 'temp_java':
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

    return java_file_counter

# Set the source directory to the current directory
src_dir = "."  # Current directory

# Get a list of directories (repos) in the source directory
repo_dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

# Process each repo folder
for repo_dir in repo_dirs:
    num_java_files = process_repo_folder(os.path.join(src_dir, repo_dir))
    print("Java files moved to temp_java folder in", repo_dir)
    print("Number of Java files moved:", num_java_files)

    # Move Java files back to the original repo folder from temp_java folder
    temp_java_dir = os.path.join(repo_dir, 'temp_java')
    for filename in os.listdir(temp_java_dir):
        src_path = os.path.join(temp_java_dir, filename)
        dst_path = os.path.join(repo_dir, filename)
        shutil.move(src_path, dst_path)

    # Remove the temp_java folder
    shutil.rmtree(temp_java_dir)

    print("Java files moved back to", repo_dir)
