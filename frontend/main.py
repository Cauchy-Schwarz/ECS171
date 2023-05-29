from flask import Flask, render_template, request, redirect, jsonify
import json
import csv

from io import BytesIO
import os
import torch 
import torch.nn as nn 
import torch.optim as optim
from torchvision import transforms, datasets, models
from PIL import Image

app = Flask(__name__)

model = torch.load('effnetLArch.pth' ,map_location = torch.device('cpu'))
model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

learning_rate = 0.001

# Loss Function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

with open('./numbers.txt') as f:
    lines = f.read().splitlines()
c = dict()
for i, x in enumerate(lines):
    c[i] = x

@app.route('/')
def index():
    return render_template('layout.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		imageRequest = request.files['file']
		print(imageRequest)
		loader = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		#image = Image.open("./car_data/car_data/test/Acura RL Sedan 2012/00183.jpg")
		image = Image.open(imageRequest)
		image = loader(image).float()
		image = torch.autograd.Variable(image, requires_grad=True)
		image = image.unsqueeze(0)
		image = image.to(device)
		output = model(image)
		conf, predicted = torch.max(output.data, 1)
		print(c[predicted.item()], "confidence: ", conf.item())
	return jsonify(c[predicted.item()])

app.run(host='0.0.0.0', port=81)
