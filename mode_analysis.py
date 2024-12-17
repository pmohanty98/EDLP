import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from collections import Counter
from torchvision import datasets
from torch.utils.data import DataLoader
import random
import glob


# Step 1: Define a simple LeNet5 model (you can use other pre-trained models as well)
class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Step 2: Set random seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Step 3: Function to load and resize images
def load_and_resize_image(img_path):
    image = Image.open(img_path).convert('L')  # Convert image to grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize (similar to MNIST data)
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


# Step 4: Training process
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    print('Training complete.')


def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on the MNIST test set: {accuracy:.2f}%')
    return accuracy
# Step 5: Function to predict classes for all images in the directory
def predict_image_class(model, image_tensor):
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output.data, 1)
    return predicted_class.item()


# Step 6: Plot histogram of predictions
def plot_class_distribution(predictions, dataset):
    counts = Counter(predictions)  # Count the occurrences of each class
    classes = list(counts.keys())
    frequencies = list(counts.values())
    print(sum(frequencies))

    plt.bar(classes, frequencies, color='skyblue')
    plt.xticks(range(10))  # Class labels (0-9)
    plt.xlabel('Class Label',fontsize=14)
    plt.ylabel('Frequency',fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.title('Distribution of MNIST Classes for '+str(dataset), fontsize=15)
    plt.savefig(str(dataset)+".png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Initialize device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(1234567)  # Set random seed

    # Step 7: Load the MNIST dataset for training
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize like MNIST
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # Step 8: Define the LeNet model, loss function, and optimizer
    model = LeNet5().to(device)
    criterion = torch.nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer

    sampler='hb-10-1'
    print("Training the model on MNIST dataset...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    accuracy = evaluate_model(model, test_loader)

    # Step 10: Load and resize images for prediction (after training)
    image_directory = 'figs/rbm_sample/mnist/'+sampler  # Specify your image directory
    predictions = []

    image_paths = glob.glob(os.path.join(image_directory, '*.png'))

    # Loop through images, resize them, and predict their class
    for img_path in image_paths:
        img_tensor = load_and_resize_image(img_path)  # Resize to 28x28 and convert to tensor
        predicted_class = predict_image_class(model,img_tensor)  # Predict class using your model
        predictions.append(predicted_class)

    # Step 11: Plot the class distribution based on predictions
    plot_class_distribution(predictions,sampler)