from flask import Flask
import torch


app = Flask(__name__)

@app.route('/')
def home():
    return 'Welcome to the PyTorch Flask app!'
