import os
import argparse
from tqdm import tqdm
import yaml as yaml
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from noise_scheduler import LinearNoiseScheduler
from model import UNET

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sample(
  model,
  scheduler,
  train_config,
  model_config,
  diffusion_config,      
):
    # create a random sample based on numer of images in train_config
    x_t = torch.randn(
        (train_config['num_samples'],
         model_config['image_channels'],
         model_config['image_size'],
         model_config['image_size']
        )
    ).to(device)    # our x_t

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        tensor_i = torch.tensor(i).to(device)
        pred_for_noisy_image = model(x_t, tensor_i.unsqueeze(0))

        # get x_0, x_(t-1) => saved int0 x_t itself like standard looping technique
        x_t, x0_pred = scheduler.sample_from_reverse(x_t, pred_for_noisy_image, tensor_i)   
        # we basically get the mean and variance similar to re-paramaerization trick, chk ./noise_scheduler.py

        # save predicted x_0
        images = torch.clamp(x_t, -1., 1.).detach().cpu()
        images = (images + 1) / 2   # re-scale
        grid = make_grid(images, nrow=train_config['num_grid_rows'])
        t1 = transforms.ToPILImage()
        images = t1(grid)
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'))
        images.save(os.path.join(train_config['task_name'], 'samples', f'x0_{i}.png'))
        images.close()

def infer(args):
    #### Setup ####
    with open(args.config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)

    print(f'[INFO]: In inference, working with config : {config}')

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # Noise Scheduler
    noise_scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end'],
    )

    # Load Model
    model = UNET(model_config).to(device)
    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        sample(model, noise_scheduler, train_config, model_config, diffusion_config)

    print("[INFO] : Finished Inference. Check folder for sampled outputs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for unconditional DDPM sampling')
    parser.add_argument("--config", dest='config_path', default='config.yaml', type=str)
    args = parser.parse_args()

    infer(args)