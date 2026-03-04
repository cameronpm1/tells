import os
import cv2
import torch
import shutil
import imageio
import numpy as np
from typing import Optional

from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader

from learn.belief.models import *
from learn.rl.custom_callbacks import create_image
from learn.belief.custom_dataset import CustomDataset

COLOR_SCALE = np.array([
    [255, 0,   0],
    [242, 13,  0],
    [228, 27,  0],
    [215, 40,  0],
    [201, 54,  0],
    [188, 67,  0],
    [174, 81,  0],
    [161, 94,  0],
    [147, 108, 0],
    [134, 121, 0],
    [120, 135, 0],
    [107, 148, 0],
    [93,  162, 0],
    [80,  175, 0],
    [67,  188, 0],
    [53,  202, 0],
    [40,  215, 0],
    [26,  229, 0],
    [13,  242, 0],
    [0,   0, 255]
], dtype=np.uint8)

def boat_plot_data(
    pos,
    hdg,
    scale=10,
    color=None,
) -> dict:
    '''
    returns dictionary w/ plottind data for Renderer2D

    output
    ------
    dict
        plotting data keys:[lines,points]
    '''

    boat_points = np.array([
        [0.0, 0.25],
        [1.5, 0.25],
        [2.0, 0.0],
        [1.5, -0.25],
        [0.0, -0.25]
    ])

    boat_lines = np.array([
        (0,1), (1,2), (2,3), (3,4), (4,0)
    ])

    points = []
    lines = []
    colors = []

    for i,(pos,hdg) in enumerate(zip(pos,hdg)):

        dcm = np.array([
            [np.cos(hdg), -np.sin(hdg)],
            [np.sin(hdg),  np.cos(hdg)]
        ])
        transformed_vertices = np.dot(boat_points*scale, dcm.T) + pos*scale
        for j in range(len(boat_points)):
            points.append(transformed_vertices[j])
            lines.append((boat_lines[j][0]+(5*i),boat_lines[j][1]+(5*i)))
            if color is not None:
                colors.append(color[i])
            else:
                colors.append('k')

    plot_data = {}
    plot_data['lines'] = lines
    plot_data['points'] = points
    plot_data['colors'] = colors

    return plot_data

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

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

class BeliefModel(pl.LightningModule):

    def __init__(self,
                 lr: float=1e-4,
                 seed: int=1,
                 if_cuda: bool=True,
                 if_test: bool=False,
                 gamma: float=1.0,
                 log_dir: str='logs',
                 train_batch: int=512,
                 val_batch: int=256,
                 test_batch: int=256,
                 num_workers: int=8,
                 model_name: str='NN2CNN',
                 noise_val: Optional[float]=None,
                 data_dir: str='data',
                 data_name: str='sinusoidal_data',
                 input_channels: int=1,
                 output_channels: int=1,
                 lr_schedule: list=[],
                 output_noise: float=0.0,
                 **kwargs,
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True} if self.hparams.if_cuda else {}
        self.__build_model()

    def __build_model(self):
        #if self.hparams.model_name == 'bayesianCNN':
        #self.model = NN2CNN(self.hparams.input_channels,self.hparams.output_channels)
        self.model = NN(self.hparams.input_channels,self.hparams.output_channels)
        self.loss_func = PermutationInvariantMSE() #torch.nn.MSELoss()

    def train_forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        data, target, filepath = batch
        output = self.train_forward(data)

        train_loss = self.loss_func(output,target)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        data, target, filepath = batch
        output = self.train_forward(data)

        val_loss = self.loss_func(output,target)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        data, target, filepath = batch
        self.all_filepaths.extend(filepath)

        error = 0

        output = self.model(data)
        #avg_error = np.sum(np.linalg.norm(output.detach().cpu().numpy() - target.squeeze().detach().cpu().numpy(),axis=1))/len(data)
        avg_error = self.loss_func.error(output.detach().cpu().numpy(), target.squeeze().detach().cpu().numpy()) / len(data)
        test_loss = self.loss_func(output,target.squeeze())
        self.log('avg_error', avg_error, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.all_predictions.extend(output.detach().cpu().numpy())
        self.all_ground_truths.extend(target.squeeze().detach().cpu().numpy())
        self.all_inputs.extend(data.detach().cpu().numpy())

    def test_save(self):
        #print(self.all_filepaths)
        folderlist = self.test_dataset.get_folderlist()
        folder_idxs = np.random.choice(len(folderlist), size=5, replace=False)
        #generate and save images
        save_dir = os.path.join(self.hparams.log_dir, 'test_outputs')
        mkdir(save_dir)
        print('Saving test output images to: ', save_dir)
        for folder_idx in folder_idxs:
            folder = folderlist[folder_idx]
            file_names = [s for i, s in enumerate(self.test_dataset.get_filelist()) if folder in s]
            if len(file_names) > 0:
                file_names.sort()
                file_idxs = [self.all_filepaths.index(s) for s in file_names if s in self.all_filepaths]
                #for i in file_idxs:
                #    print(self.all_filepaths[i])
                imgs = []
                for idx in file_idxs:
                    pred = self.all_predictions[idx]
                    gt = self.all_ground_truths[idx]
                    start = self.all_inputs[idx]
                    poses = np.array([
                        [0.0,0.0],
                        [start[3],start[4]],
                        [gt[0],gt[1]],
                        [gt[3],gt[4]],
                        [pred[0],pred[1]],
                        [pred[3],pred[4]],
                    ])
                    hdgs = np.array([
                        start[2],
                        start[7],
                        gt[2],
                        gt[5],
                        pred[2],
                        pred[5],
                    ])
                    #print(np.linalg.norm(np.array([gt[0],gt[1]]) - np.array([pred[0],pred[1]])))
                    clrs = ['k','k','g','g','r','r']
                    plot_data = boat_plot_data(poses, hdgs, scale=10, color=clrs)
                    img = create_image(plot_data,xlim=(-300,300),ylim=(-300,300))
                    imgs.append(img)

                save_file = os.path.join(save_dir, f'sim_{folder_idx}.gif')
                save_cv2_images_as_gif(imgs, save_file, fps=0.001)
                #cv2.imwrite(save_file, img)
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def setup(self, stage=None):
        if stage=='fit':
            kwargs = {
                    'data_dir':self.hparams.data_dir,
                    'data_name':self.hparams.data_name,
                    'flag':'train',
                    'seed':self.hparams.seed,
                }
            self.train_dataset = CustomDataset(**kwargs)
            kwargs['flag'] = 'val'
            self.val_dataset = CustomDataset(**kwargs)
        if stage=='test':
            kwargs = {
                    'data_dir':self.hparams.data_dir,
                    'data_name':self.hparams.data_name,
                    'flag':'test',
                    'seed':self.hparams.seed,
                }
            self.test_dataset = CustomDataset(**kwargs)
            self.all_predictions = []
            self.all_ground_truths = []
            self.all_inputs = []
            self.all_filepaths = []
            self.all_sims = []
        pass

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.hparams.train_batch,
                                                   shuffle=True,
                                                   **self.kwargs)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                 batch_size=self.hparams.val_batch,
                                                 shuffle=False,
                                                 **self.kwargs)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.hparams.test_batch,
                                                  shuffle=False,
                                                  **self.kwargs)
        return test_loader