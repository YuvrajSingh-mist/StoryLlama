
# download_model_weight.py
import os
import argparse
from huggingface_hub import hf_hub_download, login

def download_model(repo_id, filename, cache_dir):

    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True,
            force_download=False,
            token=os.getenv("HF_TOKEN")
        )
       
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1024*1024:
            return model_path
        raise ValueError("Downloaded file is too small or invalid")
    except Exception as e:
        print(f"Download failed: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download models from Hugging Face Hub")
    parser.add_argument("--model_type", 
                      choices=["sft", "dpo", "pretrained"],
                      required=True,
                      help="Type of model to download")
    
    args = parser.parse_args()

    model_config = {
        "pretrained": {
            "repo_id": "YuvrajSingh9886/smol-llama-base",
            "filename": "snapshot_6750.pt",
            "cache_dir": "weights/pretrained"
        }
    }

    config = model_config[args.model_type]
    os.makedirs(config["cache_dir"], exist_ok=True)
    
    print(f"Downloading {args.model_type} model...")
    model_path = download_model(
        config["repo_id"],
        config["filename"],
        config["cache_dir"]
    )
    print(f"Successfully downloaded to: {model_path}")

if __name__ == "__main__":

    login(token=os.getenv("HF_TOKEN"))
    main()
