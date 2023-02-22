# accelerate launch main.py
# clearml-task --project "Text Classification" --name dreambooth --script main.py --queue "<=12GB" --args epochs=20 lr=0.005
# clearml accelerate launch main.py


#do not use clearml-task, just call the code...use task init
# from accelerate import Accelerator
# accelerator = Accelerator()
# print(123)

from clearml import Task

# note: you need "clearml==0.17.6rc1"
Task.add_requirements("./diffusers")
Task.add_requirements("triton")
Task.add_requirements("ftfy")
Task.add_requirements("safetensors")
Task.add_requirements("accelerate")
Task.add_requirements("transformers")
Task.add_requirements("bitsandbytes", "0.35.0")
Task.add_requirements("natsort")
Task.add_requirements("torchvision")
Task.add_requirements("xformers", "0.0.17.dev447")
Task.add_requirements("matplotlib")
Task.add_requirements("torch", "1.13.1+cu116")


task = Task.init(
    project_name='Text Classification',
    task_name='dreambooth',
    tags="just a test",
    auto_connect_arg_parser=True,
)

task.execute_remotely(queue_name="<=48GB", clone=False, exit_process=True)

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_name_or_path', type=str, default="runwayml/stable-diffusion-v1-5",required=True, help='Name or path to the pretrained model')
parser.add_argument('--pretrained_vae_name_or_path', type=str, default="stabilityai/sd-vae-ft-mse", required=True, help='Name or path to the pretrained VAE model')
parser.add_argument('--output_dir', type=str, default="content/stable_diffusion_weights/bfn", required=True, help='Output directory to save model and samples')
parser.add_argument('--revision', type=str, default='fp16', help='Revision')
parser.add_argument('--with_prior_preservation', action='store_true', help='Use prior preservation')
parser.add_argument('--prior_loss_weight', type=float, default=1.0, help='Weight for prior loss')
parser.add_argument('--seed', type=int, default=1337, help='Seed for reproducibility')
parser.add_argument('--use_8bit_adam', action='store_true', help='Use 8-bit Adam optimizer')
parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images')
parser.add_argument('--train_batch_size', type=int, default=1, help='Training batch size')
parser.add_argument('--train_text_encoder', action='store_true', help='Train text encoder')
parser.add_argument('--mixed_precision', type=str, default='fp16', help='Mixed precision mode')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
parser.add_argument('--lr_scheduler', type=str, default='constant', help='Learning rate scheduler')
parser.add_argument('--lr_warmup_steps', type=int, default=0, help='Number of warmup steps for the learning rate scheduler')
parser.add_argument('--num_class_images', type=int, default=50, help='Number of images for class embedding')
parser.add_argument('--sample_batch_size', type=int, default=4, help='Batch size for sampling during training')
parser.add_argument('--max_train_steps', type=int, default=800, help='Maximum number of training steps')
parser.add_argument('--save_interval', type=int, default=10000, help='Interval to save checkpoints')
parser.add_argument('--save_sample_prompt', type=str, default='photo of bfn bear', help='Prompt to generate samples')
parser.add_argument('--concepts_list', type=str, default='concepts_list.json', help='Path to concepts list file')


parser.add_argument('--object_name', type=str, default='bfn', help='Name of the object')
parser.add_argument('--model_name', type=str, default='runwayml/stable-diffusion-v1-5', help='Name or path to the pretrained model')
parser.add_argument('--output_dir', type=str, default='content/', help='Output directory to save model and samples')
parser.add_argument('--input_image_dir', type=str, default='images', help='Input image folder')

args = parser.parse_args()


#@markdown Object name used in the prompt, e.g., "a photo of [object_name] person".
OBJECT_NAME = args.object_name

#@markdown Name/Path of the initial model.
MODEL_NAME = args.model_name

#@markdown Enter the directory name to save model at.

OUTPUT_DIR = "stable_diffusion_weights/" + OBJECT_NAME #@param {type:"string"}

INPUT_IMAGES_DIR = args.input_image_dir

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[*] Weights will be saved at {OUTPUT_DIR}")

# You can also add multiple concepts here. Try tweaking `--max_train_steps` accordingly.

concepts_list = [
#    {
#        "instance_prompt":      "photo of zwx dog",
#        "class_prompt":         "photo of a dog",
#        "instance_data_dir":    "./content/data/zwx",
#       "class_data_dir":       "./content/data/dog"
#    },
     {
         "instance_prompt":      f"photo of {OBJECT_NAME} bear",
         "class_prompt":         "photo of a bear",
         "instance_data_dir":    f"content/data/{OBJECT_NAME}",
         "class_data_dir":       "content/data/bear"
     }
]

# `class_data_dir` contains regularization images
import json
import os
for c in concepts_list:
    print(c["instance_data_dir"])
    os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)


import os

import shutil

for c in concepts_list:
    print(f"Uploading instance images for `{c['instance_prompt']}`")

    #filenames = [os.path.join("pikachu", f) for f in os.listdir("pikachu") if f[-3:] == "jpg"]

    for filename in os.listdir(INPUT_IMAGES_DIR):
        print(filename)
        filename_full = os.path.join(INPUT_IMAGES_DIR, filename)
        dst_path = os.path.join(c['instance_data_dir'], filename)
        shutil.copy(filename_full, dst_path)
#subprocess

import subprocess

command = ['accelerate', 'launch', 'train_dreambooth.py',
           '--pretrained_model_name_or_path=' + MODEL_NAME,
           '--pretrained_vae_name_or_path=stabilityai/sd-vae-ft-mse',
           '--output_dir=' + OUTPUT_DIR,
           '--revision=fp16',
           '--with_prior_preservation', '--prior_loss_weight=1.0',
           '--seed=1337',
           '--use_8bit_adam',
           '--resolution=512',
           '--train_batch_size=1',
           '--train_text_encoder',
           '--mixed_precision=fp16',
           '--gradient_accumulation_steps=1',
           '--learning_rate=1e-6',
           '--lr_scheduler=constant',
           '--lr_warmup_steps=0',
           '--num_class_images=50',
           '--sample_batch_size=4',
           '--max_train_steps=800',
           '--save_interval=10000',
           '--save_sample_prompt=photo of bfn bear',
           '--concepts_list=concepts_list.json']

subprocess.run(command)

WEIGHTS_DIR = "" #@param {type:"string"}
if WEIGHTS_DIR == "":
    from natsort import natsorted
    from glob import glob
    import os
    WEIGHTS_DIR = natsorted(glob(OUTPUT_DIR + os.sep + "*"))[-1]
print(f"[*] WEIGHTS_DIR={WEIGHTS_DIR}")

# @markdown Run to generate a grid of preview images from the last saved weights.
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

weights_folder = OUTPUT_DIR
folders = sorted([f for f in os.listdir(weights_folder) if f != "0"], key=lambda x: int(x))

row = len(folders)
print(folders[0])
col = len(os.listdir(os.path.join(weights_folder, folders[0], "samples")))
scale = 4
fig, axes = plt.subplots(row, col, figsize=(col * scale, row * scale), gridspec_kw={'hspace': 0, 'wspace': 0})

for i, folder in enumerate(folders):
    folder_path = os.path.join(weights_folder, folder)
    image_folder = os.path.join(folder_path, "samples")
    images = [f for f in os.listdir(image_folder)]
    for j, image in enumerate(images):
        if row == 1:
            currAxes = axes[j]
        else:
            currAxes = axes[i, j]
        if i == 0:
            currAxes.set_title(f"Image {j}")
        if j == 0:
            currAxes.text(-0.1, 0.5, folder, rotation=0, va='center', ha='center', transform=currAxes.transAxes)
        image_path = os.path.join(image_folder, image)
        img = mpimg.imread(image_path)
        currAxes.imshow(img, cmap='gray')
        currAxes.axis('off')

plt.tight_layout()
plt.savefig('grid.png', dpi=72)


import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from IPython.display import display

model_path = WEIGHTS_DIR # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive

pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
g_cuda = None

#@markdown Can set random seed here for reproducibility.
g_cuda = torch.Generator(device='cuda')
seed = 52362 #@param {type:"number"}
g_cuda.manual_seed(seed)


#@title Run for generating images.

prompt = f"photo of {OBJECT_NAME} bear cooking" #@param {type:"string"}
negative_prompt = "" #@param {type:"string"}
num_samples = 4 #@param {type:"number"}
guidance_scale = 7.5 #@param {type:"number"}
num_inference_steps = 24 #@param {type:"number"}
height = 512 #@param {type:"number"}
width = 512 #@param {type:"number"}

with autocast("cuda"), torch.inference_mode():
    images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g_cuda
    ).images

for img in images:
    display(img)