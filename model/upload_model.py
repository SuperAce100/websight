#!/usr/bin/env python3
"""
Upload ui-tars fine-tuned model to Hugging Face Hub
"""

from huggingface_hub import HfApi, upload_file
import os

def upload_model():
    # Initialize the API
    api = HfApi()
    
    # Configuration
    model_file = "ui-tars_finetuned.safetensors"
    repo_name = "tanvirb/ui-tars-finetuned"  # Change this to your desired repo name
    
    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"❌ Model file {model_file} not found!")
        return
    
    try:
        # Create repository (if it doesn't exist)
        print(f"Creating repository: {repo_name}")
        api.create_repo(
            repo_id=repo_name,
            exist_ok=True,  # Don't fail if repo already exists
            repo_type="model"
        )
        
        # Upload the model file
        print(f"Uploading {model_file}...")
        upload_file(
            path_or_fileobj=model_file,
            path_in_repo="model.safetensors",  # Standard name for safetensors
            repo_id=repo_name,
            repo_type="model"
        )
        
        print(f"✅ Successfully uploaded to https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    upload_model() 