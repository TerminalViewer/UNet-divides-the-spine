import numpy as np
import os
import fun
import glob
import paddle.nn.functional as F
from PIL import Image
import nibabel as nib
import paddle
from tqdm import tqdm
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.vision.transforms as T
from paddle.io import Dataset
import matplotlib.pyplot as plt

# set folder paths for train and validation data
data_folder_path = "D:/VR/UNet分割脊柱/data"
train_data = data_folder_path + "/dataset-verse19training/"
validation_data = data_folder_path + "/dataset-verse19validation/"

# 数据理解

# get one image to load
train_data_raw_image = train_data + "/rawdata/sub-verse004/sub-verse004_ct.nii.gz"

one_image = nib.load(train_data_raw_image)

# look at image shape
print(one_image.shape)

# look at image header. To understand header please refer to: https://brainder.org/2012/09/23/the-nifti-file-format/
print(one_image.header)
#
# look at the raw data
one_image_data = one_image.get_fdata()
print(one_image_data)

# Visualize one image in three different angles
one_image_data_axial = one_image_data

# change the view
one_image_data_sagittal = np.transpose(one_image_data, [2, 1, 0])
one_image_data_sagittal = np.flip(one_image_data_sagittal, axis=0)

# change the view
one_image_data_coronal = np.transpose(one_image_data, [2, 0, 1])
one_image_data_coronal = np.flip(one_image_data_coronal, axis=0)

fig, ax = plt.subplots(1, 3, figsize=(60, 60))
ax[0].imshow(one_image_data_axial[:, :, 10], cmap='bone')
ax[0].set_title("Axial view", fontsize=60)
ax[1].imshow(one_image_data_sagittal[:, :, 150], cmap='bone')
ax[1].set_title("Sagittal view", fontsize=60)
ax[2].imshow(one_image_data_coronal[:, :, 150], cmap='bone')
ax[2].set_title("Coronal view", fontsize=60)
plt.show()

# Overlay a mask on top of raw image (one slice of CT-scan)
train_data_mask_image = train_data + "/derivatives/sub-verse007/sub-verse007_seg-vert_msk.nii.gz"
train_data_mask_image = nib.load(train_data_mask_image).get_fdata()

plt.figure(figsize=(10, 10))

rotated_raw = np.transpose(one_image_data, [2, 1, 0])
rotated_raw = np.flip(rotated_raw, axis=0)
plt.imshow(rotated_raw[:, :, 150], cmap='bone', interpolation='none')

train_data_mask_image[train_data_mask_image == 0] = np.nan
rotated_mask = np.transpose(train_data_mask_image, [2, 1, 0])
rotated_mask = np.flip(rotated_mask, axis=0)
plt.imshow(rotated_mask[:, :, 150], cmap='cool')

# 预处理数据

# Set paths to store processed train and validation raw images and masks
processed_train = "./processed_train/"
processed_validation = "./processed_validation/"

processed_train_raw_images = processed_train + "raw_images/"
processed_train_masks = processed_train + "masks/"

processed_validation_raw_images = processed_validation + "raw_images/"
processed_validation_masks = processed_validation + "masks/"

# Read all 2019 and 2020 raw files, both train and validation
raw_train_files = glob.glob(os.path.join(train_data, 'rawdata/*/*nii.gz'))
raw_validation_files = glob.glob(os.path.join(validation_data, 'rawdata/*/*nii.gz'))
print("Raw images count train: {0}, validation: {1}".format(len(raw_train_files), len(raw_validation_files)))

# Read all 2019 and 2020 derivatives files, both train and validation
masks_train_files = glob.glob(os.path.join(train_data, 'derivatives/*/*nii.gz'))
masks_validation_files = glob.glob(os.path.join(validation_data, 'derivatives/*/*nii.gz'))
print("Masks images count train: {0}, validation: {1}".format(len(masks_train_files), len(masks_validation_files)))

# Loop over raw images and masks and generate 'PNG' images.
print("Processing started.")
for each_raw_file in raw_train_files:
    raw_file_name = each_raw_file.split("\\")[-1].split("_ct.nii.gz")[0]
    for each_mask_file in masks_train_files:
        if raw_file_name in each_mask_file.split("\\")[-1]:
            fun.generate_data(each_raw_file,
                          each_mask_file,
                          raw_file_name,
                          processed_train_raw_images,
                          processed_train_masks)
print("Processing train data done.")

# Loop over raw images and masks and generate 'PNG' images.
for each_raw_file in raw_validation_files:
    raw_file_name = each_raw_file.split("\\")[-1].split("_ct.nii.gz")[0]
    for each_mask_file in masks_validation_files:
        if raw_file_name in each_mask_file.split("\\")[-1]:
            fun.generate_data(each_raw_file,
                          each_mask_file,
                          raw_file_name,
                          processed_validation_raw_images,
                          processed_validation_masks)
print("Processing validation data done.")
#
# Define model parameters
DEVICE = "gpu" if paddle.is_compiled_with_cuda() else "cpu"

# image size to convert to
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 250
LEARNING_RATE = 1e-4
BATCH_SIZE = 10
EPOCHS = 1
NUM_WORKERS = 8

# Set the device
paddle.device.set_device(DEVICE)

class DoubleConv(nn.Layer):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2D(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(mid_channels),
            nn.ReLU(),
            nn.Conv2D(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Layer):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2D(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Layer):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.Conv2DTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2, bias_attr=bool)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = paddle.concat([x2, x1], axis=1)
        return self.conv(x)

class OutConv(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Defining UNet architecture
# Source code: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
class UNet(nn.Layer):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Define PaddlePaddle dataset class
class VerSeDataset(Dataset):
    def __init__(self, raw_images_path, masks_path, images_name):
        self.raw_images_path = raw_images_path
        self.masks_path = masks_path
        self.images_name = images_name

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, index):
        # get image and mask for a given index
        img_path = os.path.join(self.raw_images_path, self.images_name[index])
        mask_path = os.path.join(self.masks_path, self.images_name[index])

        # read the image and mask
        image = Image.open(img_path)
        mask = Image.open(mask_path)

        # resize image and change the shape to (1, image_width, image_height)
        w, h = image.size
        image = image.resize((w, h), resample=Image.BICUBIC)
        image = T.Resize(size=(IMAGE_WIDTH, IMAGE_HEIGHT))(image)
        image_ndarray = np.asarray(image)
        image_ndarray = image_ndarray.reshape(1, image_ndarray.shape[0], image_ndarray.shape[1])

        # resize the mask. Mask shape is (image_width, image_height)
        mask = mask.resize((w, h), resample=Image.NEAREST)
        mask = T.Resize(size=(IMAGE_WIDTH, IMAGE_HEIGHT))(mask)
        mask_ndarray = np.asarray(mask)

        return {
            'image': paddle.to_tensor(image_ndarray.copy()).astype('float32'),
            'mask': paddle.to_tensor(mask_ndarray.copy()).astype('float32')
        }

class DataLoader(paddle.io.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 collate_fn=None,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False

        super().__init__(
            dataset,
            feed_list=None,
            places=None,
            return_list=return_list,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=True,
            use_shared_memory=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn)
        if sampler is not None:
            self.batch_sampler.sampler = sampler
# Get path for all images and masks
train_images_paths = os.listdir(processed_train_raw_images)
train_masks_paths = os.listdir(processed_train_masks)

validation_images_paths = os.listdir(processed_validation_raw_images)
validation_masks_paths = os.listdir(processed_validation_masks)

# Load both images and masks data
train_data = VerSeDataset(processed_train_raw_images, processed_train_masks, train_images_paths)
valid_data = VerSeDataset(processed_validation_raw_images, processed_validation_masks, validation_images_paths)

# Create PaddlePaddle DataLoader
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)



next_image = next(iter(valid_dataloader))
fig, ax = plt.subplots(1, 2, figsize=(60, 60))
ax[0].imshow(next_image['image'][0][0, :, :], cmap='bone')
ax[0].set_title("Raw image", fontsize=60)
ax[1].imshow(next_image['mask'][0][:, :], cmap='bone')
ax[1].set_title("Mask image", fontsize=60)
plt.show()


# Define Dice loss class
class DiceLoss(nn.Layer):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = paddle.nn.functional.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = paddle.reshape(inputs, [-1])
        targets = paddle.reshape(targets, [-1])

        intersection = paddle.sum(inputs * targets)
        dice = (2.0 * intersection + smooth) / (paddle.sum(inputs) + paddle.sum(targets) + smooth)

        bce = paddle.nn.functional.binary_cross_entropy_with_logits(inputs, targets)
        pred = paddle.nn.functional.sigmoid(inputs)
        loss = bce * 0.5 + dice * (1 - 0.5)

        # Subtract 1 to calculate loss from dice value
        return 1 - dice

# Define model as UNet
model = UNet(n_channels=1, n_classes=1)
model.to(device=DEVICE)

# Define the optimizer
optimizer = opt.Adam(parameters=model.parameters(), learning_rate=LEARNING_RATE)

# Train and validate
train_loss = []
val_loss = []

for epoch in range(EPOCHS):
    model.train()
    train_running_loss = 0.0
    counter = 0

    with tqdm(total=len(train_data), desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
        for batch in train_dataloader:
            counter += 1
            image = batch['image']
            mask = batch['mask']

            optimizer.clear_grad()
            outputs = model(image)
            outputs = outputs.squeeze(1)
            loss = DiceLoss()(outputs, mask)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar.update(image.shape[0])
            pbar.set_postfix(**{'loss (batch)': loss.item()})
        train_loss.append(train_running_loss / counter)

    model.eval()
    valid_running_loss = 0.0
    counter = 0
    with paddle.no_grad():
        for i, data in enumerate(valid_dataloader):
            counter += 1

            image = data['image']
            mask = data['mask']
            outputs = model(image)
            outputs = outputs.squeeze(1)

            loss = DiceLoss()(outputs, mask)
            valid_running_loss += loss.item()

        val_loss.append(valid_running_loss / counter)

# Plot train vs validation loss
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color="orange", label='train loss')
plt.plot(val_loss, color="red", label='validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Save the trained model
paddle.save({
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "./unet_model.pdparams")

# Visually look at one prediction
next_image = next(iter(valid_dataloader))

# do predict
outputs = model(next_image['image'].float())
outputs = outputs.detach().cpu()
loss = DiceLoss()(outputs, next_image['mask'])
print("Dice Score: ", 1 - loss.item())
outputs[outputs <= 0.0] = 0
outputs[outputs > 0.0] = 1.0

# plot all three images
fig, ax = plt.subplots(1, 3, figsize=(60, 60))
ax[0].imshow(next_image['image'][0][0, :, :], cmap='bone')
ax[0].set_title("Raw Image", fontsize=60)
ax[1].imshow(next_image['mask'][0][0, :, :], cmap='bone')
ax[1].set_title("True Mask", fontsize=60)
ax[2].imshow(outputs[0, 0, :, :], cmap='bone')
ax[2].set_title("Predicted Mask", fontsize=60)
plt.show()
