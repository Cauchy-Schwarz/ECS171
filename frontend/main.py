from flask import Flask, render_template, request, redirect, jsonify
import json
import csv

from io import BytesIO
import os
import torch 
import torch.nn as nn 
import torch.optim as optim
from torchvision import transforms, datasets, models
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as T
from PIL import Image

app = Flask(__name__)
#EfficientNet Large
model1 = torch.load('effnetLArch.pth' ,map_location = torch.device('cpu'))
model1.eval()
#EfficientNet Medium
model2 = torch.load('effnetMArch.pth' ,map_location = torch.device('cpu'))
model2.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

learning_rate = 0.001

# Loss Function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate, momentum=0.9)

lrscheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='max', patience=3, threshold = 0.9)

optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate, momentum=0.9)

lrscheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='max', patience=3, threshold = 0.9)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model1.to(device)
model2.to(device)

#Array of all the possible car makes 
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
		image = Image.open(imageRequest)
		image = loader(image).float()
		image = torch.autograd.Variable(image, requires_grad=True)
		image = image.unsqueeze(0)
		image = image.to(device)
		output1 = model1(image)
		output2 = model2(image)
		conf1, predicted1 = torch.max(output1.data, 1)
		conf2, predicted2 = torch.max(output2.data, 1)
		print(c[predicted1.item()], "confidence: ", conf1.item())
		print(c[predicted2.item()], "confidence: ", conf2.item())
	return jsonify(c[predicted1.item()], c[predicted2.item()])

app.run(host='0.0.0.0', port=81)
