# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Import all the required libraries (PyTorch, TorchVision, NumPy, Matplotlib, etc.).

### STEP 2: Download and preprocess the MNIST dataset using transforms.

### STEP 3: Create a CNN model with convolution, pooling, and fully connected layers.
### STEP 4: Set the loss function and optimizer. Move the model to GPU if available.
### STEP 5: Train the model using the training dataset for multiple epochs.

### STEP 6: Evaluate the model using the test dataset and visualize the results (accuracy, confusion matrix, classification report, sample prediction).





## PROGRAM

### Name:GANJI MUNI MADHURI

### Register Number:212223230060

```python
class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        #Include your code here

    def forward(self, x):
        #Include your code here



# Initialize the Model, Loss Function, and Optimizer
model =
criterion =
optimizer =

def train_model(model, train_loadr, num_epochs=10):
    #Include your code here

```

### OUTPUT

## Training Loss per Epoch

<img width="459" height="725" alt="image" src="https://github.com/user-attachments/assets/245c8756-5351-4d82-b267-235f00199169" />


## Confusion Matrix

<img width="506" height="250" alt="image" src="https://github.com/user-attachments/assets/44994855-5453-4a94-86de-554bb3f22add" />


## Classification Report
Include classification report here

### New Sample Data Prediction
<img width="463" height="461" alt="image" src="https://github.com/user-attachments/assets/1c42bba8-a35b-49a7-87a9-b842bd7157ce" />


## RESULT
Include your result here
