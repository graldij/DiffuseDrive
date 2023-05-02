## DiffuseDrive Project Spring Semester 2023 

1. If you want to run it, you need change the output_dir and the path for loading dataset.

2. Free to change the TrainingConfig as well as the Unet parameters.

3. Use DDIMPipeline to diffuse images in latent space. The UNet has now 3 blocks for down- and upsampling because the input size decreases to 16 if img_size = 128. *small bug in code*: The output of DDIMPipeline already converts the tensor to PIL images, so if the latent_channel of autoencoder > 4, the pipeline is unrunnable.