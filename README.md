DiffuseDrive Project Spring Semester 2023

If you want to run it, you need change the output_dir and the path for loading dataset.

UNet has 3 blocks for down- and upsampling because the input size decreases to 16 if img_size = 128.

Still use DDIM pipeline, so need use vae decoder to reconstruct images.

**Bug**: In evaluate function, the input for vae.decode is not correct.