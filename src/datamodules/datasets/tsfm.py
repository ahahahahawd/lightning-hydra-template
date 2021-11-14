import albumentations as A

import torchvision.transforms as transforms


class tsfmTrainFakeSnow(object):
    """transform to train fake snow dataset from DesnowGAN"""

    def __init__(
        self,
        crop_size=128,
    ):
        self.crop_size = crop_size

    def __call__(
        self,
        inputs,
    ):
        tsfm = A.Compose([
            A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
            A.RandomScale((0.3, 1.75), p=1),
            A.RandomCrop(
                width=self.crop_size,
                height=self.crop_size,
                p=1
            ),
        ])

        tsfm_res = tsfm(image=inputs[0], masks=inputs)['masks']

        tsfm_res = list(
            map(
                lambda x: transforms.ToTensor()(x), tsfm_res
            )
        )
        return tsfm_res


class tsfmValFakeSnow(object):
    """transforms to validate fake snow dataset"""

    def __init__(
        self,
        crop_size=128,
    ):
        self.crop_size = crop_size

    def __call__(
        self,
        inputs,
    ):
        tsfm = A.Compose([
            A.RandomCrop(
                width=self.crop_size,
                height=self.crop_size,
                p=1
            ),
        ])

        tsfm_res = tsfm(image=inputs[0], masks=inputs)['masks']

        tsfm_res = list(
            map(
                lambda x: transforms.ToTensor()(x), tsfm_res
            )
        )
        return tsfm_res


class tsfmTestFakeSnow(object):
    """transforms to test fake snow dataset"""

    def __init__(
        self,
        resize_size: int = -1,
    ):
        self.resize_size = resize_size

    def __call__(
        self,
        inputs,
    ):
        # for test fake snow dataset
        if self.resize_size == -1:
            tsfm_res = inputs
        
        # for test speed when reszie to a fixed size
        else:
            tsfm = A.Compose([
                A.Resize(
                    height=self.resize_size,
                    width=self.resize_size,
                    p=1
                ),
            ])

            tsfm_res = tsfm(image=inputs[0], masks=inputs)['masks']

        tsfm_res = list(
            map(
                lambda x: transforms.ToTensor()(x), tsfm_res
            )
        )
        
        return tsfm_res


# class tsfmTestRealSnow(object):
#     """transforms to test real snow dataset"""

#     def __init__(
#         self,
#         resize_size: int = -1,
#     ):
#         self.resize_size = resize_size

#     def __call__(
#         self,
#         x,
#     ):
#         # for test real snow dataset
#         if self.resize_size == -1:
#             tsfm_res = x
        
#         # for test speed when reszie to a fixed size
#         else:
#             tsfm_res = transforms.Resize(self.resize_size)(x)

#         tsfm_res = transforms.ToTensor()(tsfm_res)
        
#         return tsfm_res


# class tsfmTrainNoiseFakeSnow(object):
    # def __init__(self,
    #              resize_size=None,
    #              crop_size=None,
    #              ):
    #     #         self.resize_size = resize_size
    #     #         self.crop_size = crop_size
    #     pass

    # def __call__(self, x1, x2, x3):
    #     tsfm_noise = A.Compose([
    #         A.GaussNoise(),
    #         # A.GaussianBlur(),
    #         # A.RandomBrightnessContrast(),
    #     ])
    #     tsfm_res = tsfm_noise(image=x1)
    #     x1 = tsfm_res['image']

    #     tsfm = A.Compose([
    #         A.HorizontalFlip(p=0.5),
    #         A.RandomScale((0.3, 1.75), p=0.5),
    #         # A.RandomBrightnessContrast(p=0.2),
    #         A.RandomCrop(width=128, height=128),
    #     ])

    #     tsfm_res = tsfm(image=x1, masks=[x2, x3])
    #     tsfm_synthetic = tsfm_res['image']
    #     tsfm_gt = tsfm_res['masks'][0]
    #     tsfm_mask = tsfm_res['masks'][1]

    #     tsfm_synthetic, tsfm_gt, tsfm_mask = map(
    #         lambda x: transforms.ToTensor()(x), [tsfm_synthetic, tsfm_gt, tsfm_mask]
    #     )
    #     return tsfm_synthetic, tsfm_gt, tsfm_mask

