from huggingface_hub import HfApi
import os
import argparse

def create_hf_space(username, repo_name, token):
    api = HfApi()
    repo_id = f"{username}/{repo_name}"
    api.create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="space",
        space_sdk="docker"
    )
    print(f"Space '{repo_id}' created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--repo_name", required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    create_hf_space(args.username, args.repo_name, args.token)
