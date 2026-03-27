import os
import cv2
import yaml
import shutil
import imageio

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def save_argb_video(images, output_path, fps=30):

    height, width, _ = images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'X264' for H.264
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in images:
        out.write(frame)

    out.release()

def save_rgb_gif(images, output_path, fps=10):

    duration = 1 / fps  # seconds per frame
    imageio.mimsave(output_path, images, duration=duration)