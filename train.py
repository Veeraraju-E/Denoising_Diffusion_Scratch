import os
from tqdm import tqdm
import yaml as yaml
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from noise_scheduler import LinearNoiseScheduler
from dataset import MNISTDataset
from model import UNET

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(args):

    #### Setup #### 
    with open(args.config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)

    print(f'[INFO]: Working with config : {config}')

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # Noise Scheduler
    noise_scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end'],
    )

    # Dataset
    mnist_dataset = MNISTDataset(
        split='train', 
        images_path=dataset_config['images_path'],
    )
    mnist_dataloader = DataLoader(
        dataset=mnist_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=4,
    )

    # Model
    model = UNET(model_config).to(device)
    model.train()

    # Output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    # Checkpoint if found
    ckpt_path = os.path.join(train_config['config_name'], train_config['ckpt_name'])
    if os.path.exists(ckpt_path):
        print(f"[INFO] : Loading checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # HParams
    num_epochs = train_config['num_epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    #### Training Loop ####
    for epoch in range(num_epochs):
        losses = []

        for images, _ in tqdm(mnist_dataloader):
            images = images.to(device)    # unconditional ddpm to start off with, therefore no labels
            optimizer.zero_grad()

            # First, generate random noise and timestamp to add to image
            noise = torch.randn_like(images).to(device)
            t = torch.randint(0, diffusion_config['num_timesteps'], (images.shape[0],)).to(device)

            noisy_images = noise_scheduler.add_noise(images, noise, t)
            pred_for_noisy_image = model(noisy_images, t)

            loss = criterion(pred_for_noisy_image, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f"Finished epoch : {epoch + 1}, avg_cum_loss : {np.mean(losses):.4f}")
        torch.save(model.state_dict(), ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for unconditional DDPM sampling')
    parser.add_argument("--config", dest='config_path', default='config.yaml', type=str)
    args = parser.parse_args()

    train(args)