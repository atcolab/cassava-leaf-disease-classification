import cv2 
import config
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

transforms_train = A.Compose([
    A.RandomResizedCrop(height=config.IMG_SIZE, width=config.IMG_SIZE, p=0.5),
    A.Transpose(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    A.CenterCrop(config.IMG_SIZE, config.IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    A.CoarseDropout(p=0.5, max_height=20, max_width=20),
    A.Cutout(p=0.5, max_h_size=20, max_w_size=20),
    ToTensorV2(),
],p=1.0)

transforms_valid = A.Compose([
    A.CenterCrop(config.IMG_SIZE, config.IMG_SIZE, p=0.5),
    A.Resize(config.IMG_SIZE, config.IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ToTensorV2(),
],p=1.0)