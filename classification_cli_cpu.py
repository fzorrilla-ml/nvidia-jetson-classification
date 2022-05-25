# IDENTIFICATION DIVISION.
# PROGRAM-ID.   ASS4-5.
# AUTHOR.       FRANCIS ZORRILLA.
# INSTALLATION. IBM-PC.
# DATE-WRITEN.  01/11/22.
# SECURITY.     CONFIDENTIAL.
#
# *     Face Liveness Detection from A Single Image with Deep Learning Model
# *     Classification CLI using CPU

import torch
import torch.nn.functional as F
import torchvision
import cv2
import face_recognition

from utils import preprocess

import time
import os
import sys

# TASK = 'screen'

CATEGORIES = ['yes', 'no']

print('\n')
print("Face Liveness Detection from A Single Image with Deep Learning Model")
print("Classification CLI using CPU")
print("PyTorch version: " + torch.__version__)

device = torch.device('cpu')
# RESNET 34
model = torchvision.models.resnet34(pretrained=True)
# Fully Connected Layer Neural Network
# Reinitialize model.fc to be a Linear layer with 512 input features and 2 output features
model.fc = torch.nn.Linear(512, len(CATEGORIES))
model = model.to(device)
model_path = 'screen_model_resnet34_gpu_epochs100.pth'
if os.path.exists(model_path):
    # Using the option map_location=device you can use a model trained in GPUs on a CPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    # You must call model.eval() to set dropout and batch normalization layers to evaluation mode
    # before running inference. Failing to do this will yield inconsistent inference results.
    model.eval()
    print("Model configured: " + model_path)
else:
    print("Model file [" + model_path + "] not found.")
    quit()
              
def inference(task, model, file_path):
    start = time.time()
    img = cv2.imread(file_path)
    # This conversion is necessary if OpenCV uses BGR instead of RGB
    # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract the region of the image that contains the face, model="hog" it is faster in CPUs. 
    face_locations = []
    face_locations = face_recognition.face_locations(img, model="hog")

    if len(face_locations) >= 1:
        # Just use the first face to make the prediction
        top, right, bottom, left = face_locations[0]
        face_image = img[top:bottom, left:right]

        # Show the image in a new Window
        cv2.imshow('face_image',face_image)
        cv2.waitKey()
      
        preprocessed = preprocess(face_image)
        output = model(preprocessed)
        output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        category_index = output.argmax()
        
        end = time.time()
        print("Time elapsed: \t" + "{:.4f}".format(end - start) + " seconds")
        print('\n')
        print("Odds of being a " + task)
        print("======================")
        for i, score in enumerate(list(output)):
            print(CATEGORIES[i]+'\t' + "{:.4f}".format(score))   
        print('\n')    
        print("Task: " + task + " prediction " + CATEGORIES[category_index] + '\n') 
    else:
        print("No faces were detected")     


def main():
    # args is a list of the command line args
    args = sys.argv[1:]
    
    if len(args) == 1:
        if os.path.exists(args[0]):
            print("Image file:\t", args[0])
            inference('screen', model, args[0])
        else:
            print("File not found.")    
    else:
        print("Invalid argument. Usage [python classification_cli_cpu.py image_file_name]")

if __name__ == "__main__":
    main()




