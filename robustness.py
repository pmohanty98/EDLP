import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchattacks import FGSM
from torchattacks.attack import Attack
from scipy.ndimage import uniform_filter

def manual_ssim(original, reconstructed, C1=1e-4, C2=9e-4):
    # Compute means
    mu_x = uniform_filter(original, size=7)
    mu_y = uniform_filter(reconstructed, size=7)

    # Compute variances and covariances
    sigma_x = uniform_filter(original ** 2, size=7) - mu_x ** 2
    sigma_y = uniform_filter(reconstructed ** 2, size=7) - mu_y ** 2
    sigma_xy = uniform_filter(original * reconstructed, size=7) - mu_x * mu_y

    # Compute SSIM
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim = numerator / denominator

    return np.mean(ssim)
def manual_psnr(original, reconstructed, max_pixel_value=1.0):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal.
        return float('inf')  # Infinite PSNR
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

def compute_psnr_ssim(original, reconstructed):
    # Assuming original and reconstructed are numpy arrays with values normalized between 0 and 1
    psnr = manual_psnr(original, reconstructed)
    ssim = manual_ssim(original, reconstructed)
    return psnr, ssim


# Define transformations (resizing to 28x28 and converting to grayscale)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale if needed
    transforms.Resize((28, 28)),  # Resize images to 28x28 (like MNIST)
    transforms.ToTensor(),  # Convert to tensor
])




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Custom dataset loader for images without labels
class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = list(self.root_dir.glob('*.png'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image


# Define the autoencoder model (unsupervised)
class SimpleAutoEncoder(nn.Module):
    def __init__(self):
        super(SimpleAutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64 * 7 * 7)
        self.fc3 = nn.Linear(64 * 7 * 7, 28 * 28)  # Reconstruct 28x28 image

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, 28, 28)  # Reshape to 28x28 image
        return x


# Train autoencoder models
def train_autoencoders(models, train_loader, device, epochs=10):
    optimizers = [torch.optim.Adam(model.parameters()) for model in models]
    for model in models:
        model.train()

    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            batch_loss = 0

            # Parallel training
            for i, model in enumerate(models):
                optimizers[i].zero_grad()

                # Forward pass
                output = model(data)

                # MSE loss
                loss = F.mse_loss(output, data)
                batch_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizers[i].step()

            if batch_idx % 100 == 0:
                avg_loss = batch_loss / len(models)
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Avg Loss: {avg_loss:.6f}')


# Test models for generalization and adversarial robustness
# Function to calculate metrics across multiple models
def test_autoencoders_across_models(models, test_loader, device, noise=False, occlusion=False):
    mse_across_models, psnr_across_models, ssim_across_models, loglike_across_models = [], [], [], []

    for model in models:
        mse_combined, psnr_combined, ssim_combined, loglike_combined = [], [], [], []
        model.eval()

        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)

                # Add noise or occlusion if needed
                if noise:
                    noise_factor = 0.1
                    data += noise_factor * torch.randn_like(data)
                    data = torch.clamp(data, 0., 1.)

                if occlusion:
                    for img in data:
                        occlusion_size = random.randint(5, 10)
                        x_occlusion = random.randint(0, img.shape[1] - occlusion_size)
                        y_occlusion = random.randint(0, img.shape[2] - occlusion_size)
                        img[:, x_occlusion:x_occlusion + occlusion_size, y_occlusion:y_occlusion + occlusion_size] = 0

                # Forward pass for each model
                output = model(data)

                # Calculate per-sample MSE and aggregate for each model
                mse = F.mse_loss(output, data, reduction='none')
                mse_per_sample = mse.mean(dim=[1, 2, 3])
                mse_combined.extend(mse_per_sample.cpu().numpy())

                # Calculate PSNR and SSIM
                for i in range(data.shape[0]):
                    psnr, ssim = compute_psnr_ssim(data[i].cpu().numpy(), output[i].cpu().numpy())
                    psnr_combined.append(psnr)
                    ssim_combined.append(ssim)

                # Log-likelihood (simplified)
                loglike = -torch.sum(output - data) / torch.numel(data)
                loglike_combined.append(loglike.item())

        # Collect mean of metrics for each model
        mse_across_models.append(np.mean(mse_combined))
        psnr_across_models.append(np.mean(psnr_combined))
        ssim_across_models.append(np.mean(ssim_combined))
        loglike_across_models.append(np.mean(loglike_combined))

    # Calculate mean and standard deviation across models
    mse_mean, mse_stderr = np.mean(mse_across_models), np.std(mse_across_models) / np.sqrt(len(mse_across_models))
    psnr_mean, psnr_stderr = np.mean(psnr_across_models), np.std(psnr_across_models) / np.sqrt(len(psnr_across_models))
    ssim_mean, ssim_stderr = np.mean(ssim_across_models), np.std(ssim_across_models) / np.sqrt(len(ssim_across_models))
    loglike_mean, loglike_stderr = np.mean(loglike_across_models), np.std(loglike_across_models) / np.sqrt(len(loglike_across_models))

    print(f'Final MSE: {mse_mean:.4f} ± {mse_stderr:.4f}')
    print(f'Final PSNR: {psnr_mean:.4f} ± {psnr_stderr:.4f}')
    print(f'Final SSIM: {ssim_mean:.4f} ± {ssim_stderr:.4f}')
    print(f'Log-Likelihood: {loglike_mean:.4f} ± {loglike_stderr:.4f}')

    return mse_mean, mse_stderr, psnr_mean, psnr_stderr, ssim_mean, ssim_stderr, loglike_mean, loglike_stderr

# Main script for training and testing
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed=1234567
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./figs/rbm_sample")
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--sampler', type=str, default='edmala')
    args = parser.parse_args()
    sampler=args.sampler

    data=args.data


    num_chains = 5
    models = []
    for i in range(num_chains):
        setup_seed(seed + i)  # Use a different seed for each model to ensure variation
        model = SimpleAutoEncoder().to(device)
        models.append(model)
    """"
    dmala_models = [SimpleAutoEncoder().to(device) for _ in range(num_chains)]
    edula_models = [SimpleAutoEncoder().to(device) for _ in range(num_chains)]
    dula_models = [SimpleAutoEncoder().to(device) for _ in range(num_chains)]
    gwg_models = [SimpleAutoEncoder().to(device) for _ in range(num_chains)]
    bg_1_models = [SimpleAutoEncoder().to(device) for _ in range(num_chains)]
    hb_10_1_models = [SimpleAutoEncoder().to(device) for _ in range(num_chains)]
    """

    # Load EDMALA and DMALA datasets for training in parallel
    dataset = UnlabeledImageDataset(root_dir='./figs/rbm_sample/' + data + '/'+sampler, transform=transform)

    """
    bg_1_dataset = UnlabeledImageDataset(root_dir='./figs/rbm_sample/' + data + '/bg-1', transform=transform)
    gwg_dataset = UnlabeledImageDataset(root_dir='./figs/rbm_sample/' + data + '/gwg', transform=transform)
    hb_10_1_dataset = UnlabeledImageDataset(root_dir='./figs/rbm_sample/' + data + '/hb-10-1', transform=transform)
    """

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    """
    bg_1_loader = DataLoader(bg_1_dataset, batch_size=64, shuffle=True)
    gwg_loader = DataLoader(gwg_dataset, batch_size=64, shuffle=True)
    hb_10_1_loader = DataLoader(hb_10_1_dataset, batch_size=64, shuffle=True)
    """


    # Load the test dataset
    if data == 'mnist':
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    elif data == 'kmnist':
        test_dataset = datasets.KMNIST(root="./data", train=False, transform=transform, download=True)
    elif data == 'emnist':
        test_dataset = datasets.EMNIST(root="./data", train=False, split='mnist', transform=transform, download=True)
    else:
        test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Train the models on EDMALA and DMALA samples respectively
    print("Training models in parallel...")
    train_autoencoders(models, loader, device, epochs=25)
    """
    train_autoencoders(gwg_models, gwg_loader, device, epochs=25)
    print("Training BG-1 models in parallel...")
    train_autoencoders(bg_1_models, bg_1_loader, device, epochs=25)
    print("Training HB-10-1 models in parallel...")
    train_autoencoders(hb_10_1_models, hb_10_1_loader, device, epochs=25)
    """

    # Testing EDMALA models
    print(data, sampler)
    print(f"Testing models on clean {data} data:")
    edmala_clean_results = test_autoencoders_across_models(models, test_loader, device)
    print(f"Testing models on noisy {data} data:")
    edmala_noisy_results = test_autoencoders_across_models(models, test_loader, device, noise=True)
    print(f"Testing models on occluded {data} data:")
    edmala_occluded_results = test_autoencoders_across_models(models, test_loader, device, occlusion=True)



    """
    # Testing BG-1 models
    print(f"Testing BG-1 models on clean {data} data:")
    bg_1_clean_results = test_autoencoders(bg_1_models, test_loader, device)
    print(f"Testing BG-1 models on noisy {data} data:")
    bg_1_noisy_results = test_autoencoders(bg_1_models, test_loader, device, noise=True)
    print(f"Testing BG-1 models on occluded {data} data:")
    bg_1_occluded_results = test_autoencoders(bg_1_models, test_loader, device, occlusion=True)


    # Testing GWG models
    print(f"Testing GWG models on clean {data} data:")
    gwg_clean_results = test_autoencoders(gwg_models, test_loader, device)
    print(f"Testing GWG models on noisy {data} data:")
    gwg_noisy_results = test_autoencoders(gwg_models, test_loader, device, noise=True)
    print(f"Testing GWG models on occluded {data} data:")
    gwg_occluded_results = test_autoencoders(gwg_models, test_loader, device, occlusion=True)


    # Testing HB-10-1 models
    print(f"Testing HB-10-1 models on clean {data} data:")
    hb_10_1_clean_results = test_autoencoders(hb_10_1_models, test_loader, device)
    print(f"Testing HB-10-1 models on noisy {data} data:")
    hb_10_1_noisy_results = test_autoencoders(hb_10_1_models, test_loader, device, noise=True)
    print(f"Testing HB-10-1 models on occluded {data} data:")
    hb_10_1_occluded_results = test_autoencoders(hb_10_1_models, test_loader, device, occlusion=True)
    """


    # Summarize all results
    """
    print("\n--- SUMMARY ---")
    print(f"EDMALA Clean: {edmala_clean_results}")
    print(f"EDMALA Noisy: {edmala_noisy_results}")
    print(f"EDMALA Occluded: {edmala_occluded_results}")
    #print(f"EDMALA FGSM: {edmala_fgsm_results}")

    print(f"DMALA Clean: {dmala_clean_results}")
    print(f"DMALA Noisy: {dmala_noisy_results}")
    print(f"DMALA Occluded: {dmala_occluded_results}")
    #print(f"DMALA FGSM: {dmala_fgsm_results}")
    """
