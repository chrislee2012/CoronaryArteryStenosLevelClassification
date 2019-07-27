from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, Blur, RandomBrightnessContrast, IAAPiecewiseAffine, Flip, OneOf, Compose, RandomGamma, 
)


def light_aug(p=0.8):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        Transpose(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    ],p=p)

def medium_aug(p=0.8):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        Transpose(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
    ],p=p)


def strong_aug(p=0.8):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        Transpose(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        
        OneOf([
            RandomBrightnessContrast(),
            RandomGamma(),
            CLAHE(),
            HueSaturationValue(p=0.3),
        ])
    ],p=p)