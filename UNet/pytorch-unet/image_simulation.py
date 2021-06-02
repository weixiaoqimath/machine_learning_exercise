import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import helper
import simulation
import sys
#================================================================Synthetic images/masks for training==============================================================

# Generate some random images. n_classes = 6
input_images, target_masks = simulation.generate_random_data(192,192,count=3)
print('The shape of input images is', input_images.shape)
print('The shape of target masks is', target_masks.shape)

# (3, 192, 192, 3), 0 and 255
print('The min and max values of the input images are {} and {}'.format(input_images.min(), input_images.max()))
# (3, 6, 192, 192), 0 and 1. Each channel corresponds to one object we'd like to detect.
print('The min and max values of the target masks are {} and {}'.format(target_masks.min(), target_masks.max()))

# Change channel-order and make 3 channels for matplot
input_images_rgb = [x.astype(np.uint8) for x in input_images]

# Map each channel (i.e. class) to each color
target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]

# Left: Input image (black and white), Right: Target mask (6ch)
helper.plot_side_by_side([input_images_rgb, target_masks_rgb])

#================================================================Prepare Dataset and DataLoader===================================================================

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]

# use the same transformations for train/val in this example
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])

train_set = SimDataset(2000, transform = trans)
val_set = SimDataset(200, transform = trans)

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 25

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}
#================================================================Check the outputs from DataLoader================================================================
import torchvision.utils

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

# Get a batch of training data
inputs, masks = next(iter(dataloaders['train']))

# Shows the shape of one batch. The input shape is (25, 3, 192, 192) and the target shape is (25, 6, 192, 192)
print(inputs.shape, masks.shape)

#plt.imshow(reverse_transform(inputs[3]))