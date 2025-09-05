import os
import wandb
from loadotenv import load_env

# this gives us access to the variables in .env file
load_env()
wandb_api_key = os.environ.get("WANDB_API_KEY")

def download_artifact():
    assert 'WANDB_API_KEY' in os.environ, "WANDB_API_KEY not found in environment variables"
    wandb.login(key=wandb_api_key)
    api = wandb.Api()
    
    wandb_org = os.environ.get("WANDB_ORG")
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_model_name = os.environ.get("WANDB_MODEL_NAME")
    wandb_model_version = os.environ.get("WANDB_MODEL_VERSION")
    
    artifact_path = f'{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}'

download_artifact()