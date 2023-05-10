from flask import Flask, render_template, request, redirect, url_for
import os
import zipfile
import shutil
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import pickle
import numpy as np
import catboost
from catboost import CatBoostClassifier

app = Flask(__name__)

# Home route, returns upload page
@app.route('/')
def upload_file():
   return render_template('upload.html')

# Function to process .java files in a given directory
def process_java_files(repo_dir):
    # Define the path to the directory where the model files are located
    model_path = "/home/ahmad/Desktop/api/models/embedder_model/checkpoint-16000/"
    # Load the model configuration
    config = AutoConfig.from_pretrained(model_path)
    # Load the model
    model = AutoModel.from_pretrained(model_path, config=config)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    sum_pooled_output = None  # Initialize sum of pooled outputs
    file_count = 0  # Initialize file count

    # Walk through all directories and files in the repo directory
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        for filename in filenames:
            if filename.endswith(".java"):  # Process only .java files
                file_path = os.path.join(dirpath, filename)

                # Read the file content
                with open(file_path, "r") as f:
                    code = f.read()

                # Tokenize the code
                tokens = tokenizer.encode(code, add_special_tokens=True, truncation=True, max_length=768)
                inputs = torch.tensor([tokens])

                # Perform inference
                with torch.no_grad():
                    outputs = model(inputs)

                # Pool the output of the model over the sequence length dimension
                # (batch_size = 1, sequence_length = undefiend, hidden_size = 768 in BErt)
                # (1, number of tokens, length of outpt)
                pooled_output = torch.tensor(outputs.last_hidden_state[0, 0, :])

                # Sum the pooled outputs
                if sum_pooled_output is None:
                    sum_pooled_output = pooled_output	
                else:
                    sum_pooled_output += pooled_output

                file_count += 1  # Increment file count

    return sum_pooled_output, file_count

# Function to process the repository directory
def process_repo_folder(repo_dir):
    temp_java_dir = os.path.join(repo_dir, 'temp_java')
    # Create temporary directory for .java files if it doesn't exist
    if not os.path.exists(temp_java_dir):
        os.makedirs(temp_java_dir)

    # Move all .java files to the temporary directory
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        for filename in filenames:
            if filename.endswith(".java"):
                src_path = os.path.join(dirpath, filename)
                dst_dir = os.path.join(temp_java_dir, os.path.relpath(dirpath, repo_dir))
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                dst_path = os.path.join(dst_dir, filename)
                shutil.move(src_path, dst_path)

    # Remove all other files and directories in the repository directory
    for item in os.listdir(repo_dir):
        item_path = os.path.join(repo_dir, item)
        if item != 'temp_java':
            if os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    # Move all .java files back to the repository directory
    for dirpath, dirnames, filenames in os.walk(temp_java_dir):
        for filename in filenames:
            src_path = os.path.join(dirpath, filename)
            # Destination directory is the original repository directory
            dst_dir = os.path.join(repo_dir, os.path.relpath(dirpath, temp_java_dir))
            # Create destination directory if it doesn't exist
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            # Destination path for the file
            dst_path = os.path.join(dst_dir, filename)
            # Move the file back to the original repository directory
            shutil.move(src_path, dst_path)

    # Remove the temporary directory
    shutil.rmtree(temp_java_dir)

# Route to handle file upload
@app.route('/uploader', methods=['GET', 'POST'])
def upload_file_handler():
    if request.method == 'POST':
        # Get the uploaded file
        f = request.files['file']
        # Save the uploaded file
        f.save(f.filename)
        # Unzip the uploaded file
        with zipfile.ZipFile(f.filename, "r") as zip_ref:
            # filename without extenstion
            zip_ref.extractall(os.path.splitext(f.filename)[0])

        # Process the unzipped repository directory
        process_repo_folder(os.path.splitext(f.filename)[0])
        # Process .java files in the unzipped repository directory
        sum_pooled_output, file_count = process_java_files(os.path.splitext(f.filename)[0])

        # Remove the zip file
        os.remove(f.filename)

        # Load PCA model
        with open('models/pca.pkl', 'rb') as f:
            pca = pickle.load(f)

        # Apply PCA on pooled output vector
        pooled_output_pca = pca.transform(np.array(sum_pooled_output).reshape(1, -1))

        # Load CatBoost model
        with open('models/trained_cb.pkl', 'rb') as f:
            model = pickle.load(f)

        # Make prediction using CatBoost model
        prediction = model.predict(pooled_output_pca)
        prediction = prediction[0][0]
        # Determine design pattern based on prediction
        if prediction == 0:
            pattern = "MVC"
        elif prediction == 1:
            pattern = "MVVM"
        elif prediction == 2:
            pattern = "MVP"
        else:
            pattern = "No design pattern detected"
        # Render the result page with the predicted design pattern and file count
        return render_template('result.html', pattern=pattern, file_count=file_count)

    # Render the upload page if the request method is not POST
    return render_template('upload.html')

# Start the Flask application
if __name__ == '__main__':
   app.run(debug = True)
