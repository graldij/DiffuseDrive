from diffusers import AutoencoderKL, UNet2DConditionModel
import torch
from torchvision import transforms
from datasets import load_dataset
from torch.nn import MSELoss
from pynvml import *
import matplotlib.pyplot as plt
import os


def load_vae(repo_id, device):
    vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae")
    vae.to(device)
    return vae

# image_size can set in config of training
def transform(dataset):
    preprocess = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomPerspective(distortion_scale=0.1, p = 0.3),
        transforms.RandomGrayscale(p = 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    )
    images = [preprocess(image.convert("RGB")) for image in dataset["image"]]
    return {"images": images}


def load_images(data_path, batch_size):
    dataset = load_dataset(data_path, split="train")
    dataset.set_transform(transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def run_vae(model, dataloader):
    criterion = MSELoss()
    loss = 0
    for i, batch in enumerate(dataloader):
        images = batch["images"]
        #batch = {k: v.to("cuda") for k, v in batch.items()}
        ## manually push images to GPU becuase here no Accelarator
        images = images.cuda()

        #reconstruct = model(images)  # use whole vae
        latent = model.encode(images).latent_dist.sample()
        # decode from latent space
        reconstruct = model.decode(latent)
        reconstruct = reconstruct.sample
        loss += criterion(images, reconstruct)
        plt.imshow(reconstruct.cpu().detach().permute(0,2,3,1).numpy().reshape(64,64,3))
        plt.savefig(f'img/{i:03d}.png')
    loss = loss / len(dataloader)
    return loss

def main():
    repo_id = "/srv/beegfs02/scratch/rl_course/data/proj-diffuse-drive/stable-diffusion-v1-4"
    image_path = "data/rgb_front"
    generator = torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get vae
    vae = load_vae(repo_id, device)

    # Load images
    images = load_images(image_path, batch_size=1)

    # Use VAE, compute reconstruction error
    loss = run_vae(model=vae, dataloader=images)
    print('loss of vae is: ', loss.cpu().detach().numpy())

if __name__ == "__main__":
    main()