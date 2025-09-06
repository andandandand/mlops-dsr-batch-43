import os
import wandb
from loadotenv import load_env
from torchvision.models import resnet18, ResNet
from torch import nn
from pathlib import Path
import torch


# this gives us access to the variables in .env file
load_env()
wandb_api_key = os.environ.get("WANDB_API_KEY")

# This is the local folder where the wandb model will be downloaded
MODELS_DIR = "../models"
MODEL_FILENAME = "best_model.pth"

os.makedirs(MODELS_DIR, exist_ok=True)

def download_artifact():
    '''Download the trained model weights from Weights & Biases'''
    assert 'WANDB_API_KEY' in os.environ, "WANDB_API_KEY not found in environment variables"
    wandb.login(key=wandb_api_key)
    api = wandb.Api()
    
    wandb_org = os.environ.get("WANDB_ORG")
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_model_name = os.environ.get("WANDB_MODEL_NAME")
    wandb_model_version = os.environ.get("WANDB_MODEL_VERSION")
    
    artifact_path = f'{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}'
    print(f"Downloading artifact from {artifact_path}")
    
    artifact = api.artifact(artifact_path, type='model')
    artifact.download(root=MODELS_DIR)

def get_raw_model() -> ResNet:
     '''Get the architecture of the model (random weights), this must match the architecture used during training'''
     architecture = resnet18(weights=None)
     architecture.fc = nn.Sequential(
         nn.Linear(in_features=512, out_features=512),
         nn.ReLU(),
         nn.Linear(in_features=512, out_features=6)
     )
     
     return architecture 
 
 
def load_model() -> ResNet:
    '''Gives us the model with the trained weights'''
    download_artifact()
    model = get_raw_model()
    
    model_state_dict_path = Path(MODELS_DIR) / MODEL_FILENAME
    model.load_state_dict(torch.load(model_state_dict_path, map_location='cpu'))
    return model
