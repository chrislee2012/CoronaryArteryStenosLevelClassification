from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, Blur, RandomBrightnessContrast, IAAPiecewiseAffine, Flip, OneOf, Compose, RandomGamma, 
)

class LightAug:
    def __init__(self, p=0.8):
        self.p = p
    def __call__(self, x):
        return Compose([HorizontalFlip(),
            VerticalFlip(),
            Transpose(),
            RandomRotate90(),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        ], p=self.p)(image=x)['image']

class MediumAug:
    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, x):
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
        ], p=self.p)(image=x)['image']

class StrongAug:
    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, x):
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
        ], p=self.p)(image=x)['image']