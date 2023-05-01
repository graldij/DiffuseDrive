from dataclasses import dataclass
from datasets import load_dataset
import torch.nn.functional as F
import os

## To Do: change output_dir, dataset directory
@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 4
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 20
    gradient_accumulation_steps = 20
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_steps = 100
    save_model_epochs = 1
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    current_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = current_dir + '/../output/full_rgb_front_unconditioned_latent/20epochs_1e-4_500wu_64_128_256_256'  # the model namy locally and on the HF Hub
    unet_in_channels = 4
    device = "cuda"
    num_inference_steps = 50

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()
    
# %%
# config.dataset_name = "huggan/smithsonian_butterflies_subset"

dataset = load_dataset("imagefolder", 
                       split="train",
                       data_dir= config.current_dir + "/../data",
                       )

# %%
dataset

# %%
import matplotlib.pyplot as plt
'''
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["image"]):
    axs[i].imshow(image)
    axs[i].set_axis_off()
fig.show()
'''
# %%
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(20),
        # transforms.RandomPerspective(distortion_scale=0.1, p = 0.3),
        # transforms.RandomGrayscale(p = 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)

# %%
'''
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["images"]):
    axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
    axs[i].set_axis_off()
fig.show()
'''
# %%
import torch

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# %% 
from diffusers import UNet2DModel, AutoencoderKL
# LATER USE UNet2DConditionModel FOR CONDITIONAL MODELS

repo_id = "/srv/beegfs02/scratch/rl_course/data/proj-diffuse-drive/stable-diffusion-v1-4"
vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae")
vae.to("cuda")
model = UNet2DModel(
    sample_size= int(config.image_size / 8),  # the target image resolution
    in_channels=4,  # the number of input channels, 3 for RGB images
    out_channels=4,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    attention_head_dim=32,
    block_out_channels=(256,512,1024),  # the number of output channes for each UNet block
    down_block_types=(
        "DownBlock2D","AttnDownBlock2D", "AttnDownBlock2D"), 
    up_block_types=(
        "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
)

from diffusers import DDPMScheduler, DDIMScheduler
# consider also DDIM, should be faster

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

import torch
from PIL import Image

# %% 
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

from diffusers.optimization import get_cosine_schedule_with_warmup

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs//config.gradient_accumulation_steps),
)

from diffusers import DDPMPipeline, DDIMPipeline, LDMSuperResolutionPipeline


import math

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid


def evaluate(config, epoch, pipeline, step, model):
    latents = torch.randn(
        (config.eval_batch_size, config.unet_in_channels, config.image_size // 8, config.image_size // 8)
    )
    latents = latents.to(config.device)
    noise_scheduler.set_timesteps(config.num_inference_steps)
    latents = latents * noise_scheduler.init_noise_sigma
    

    noise_scheduler.set_timesteps(config.num_inference_steps)

    for t in tqdm(noise_scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = latents

        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = model(latent_model_input, t).sample

        # perform guidance
        noise_pred_uncond = noise_pred
        noise_pred = noise_pred_uncond

        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        ## Transform back
    
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = image - image.min()
        image = image / image.max()
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        image_grid = make_grid(pil_images, rows=1, cols=4)

        # Save the images
        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        print("saving at"+str(test_dir))
        image_grid.save(f"{test_dir}/{epoch:02d}_{step:06d}.png")
    
    
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from pathlib import Path
import os

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, vae):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="wandb",
        # logging_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("LatentDiffuseUncondition")
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images_b = batch['images']
            clean_images_dist = vae.encode(clean_images_b)
            # Sample noise to add to the images
            ## latent_dist.sample().detach()
            ## Solution found in https://github.com/huggingface/diffusers/issues/435
            clean_images = clean_images_dist.latent_dist.sample()
            noise = torch.randn(clean_images.detach().shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, len(noise_scheduler), (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # check if should evaluate the model
            if accelerator.is_main_process:
                # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                if(step%config.save_image_steps==0 and not (step==0 and epoch==0)):
                    evaluate(config, epoch, pipeline, step, model)
                    
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            # pipeline = LDMSuperResolutionPipeline(vqvae=vae, unet=model, scheduler=noise_scheduler)

            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            #     evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir) 
                    
# %%
from accelerate import notebook_launcher
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, vae)

notebook_launcher(train_loop, args, num_processes=1) 
# %%
import glob
sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])