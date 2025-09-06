import torch
import io

# This adds type hints and checking to our data 
from pydantic import BaseModel 

from fastapi import FastAPI, File, UploadFile, Depends

# This is a data model that describes the output of the API
class Result(BaseModel):
    category: str
    confidence: float
    

# Create the FastAPI instance
app = FastAPI() 

# Debug message to check that the app is running
@app.get('/')
def read_root():
    return {"message": "API is running. Visit /docs for the Swagger API documentation"}
       