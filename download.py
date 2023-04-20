import requests
import time
import os
from tqdm import tqdm

GITHUB_API = "https://api.github.com/search/repositories"
QUERY = "android+language:java"
# Alternative query
# QUERY = "android+language:java+topic:android-architecture"
DELAY = 10  # Delay in seconds between requests

headers = {
    "Accept": "application/vnd.github+json",
}

def download_repo(repo_url, destination_folder):
    if not os.path.exists(destination_folder):
        os.system(f"git clone {repo_url} {destination_folder}")
    else:
        print(f"Folder {destination_folder} already exists. Skipping download.")

def main():
    page = 1
    has_next_page = True

    while has_next_page:
        params = {
            "q": QUERY,
            "page": page,
            "per_page": 30
        }

        response = requests.get(GITHUB_API, headers=headers, params=params)

        if response.status_code == 200:
            repo_data = response.json()
            repositories = repo_data["items"]

            if not repositories:
                has_next_page = False
                break

            for repo in tqdm(repositories, desc=f"Page {page}"):
                print(f"Cloning repository: {repo['full_name']}")
                download_repo(repo["clone_url"], repo["name"])
                time.sleep(DELAY)
        else:
            print(f"Error: {response.status_code}")
            break

        page += 1

if __name__ == "__main__":
    main()
