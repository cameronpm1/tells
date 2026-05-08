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

def save_cv2_images_as_gif(images, output_path, fps=10):
    """
    images: list of cv2 images (BGR numpy arrays)
    output_path: path to save the gif (e.g., "output.gif")
    fps: frames per second
    """

    if len(images) == 0:
        raise ValueError("Image list is empty")

    rgb_frames = []

    for img in images:
        if img is None:
            continue

        # Convert BGR to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_frames.append(rgb)

    duration = 1 / fps

    imageio.mimsave(output_path, rgb_frames, duration=duration)

