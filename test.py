

from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.core import datamodule
from torchvision import transforms

from src.models.unet_resr_model import UNetResRLitModel
from src.datamodules.snow_datamodule import PairedDataModule

def predict():
    """Example of inference with trained model.
    It loads trained image classification model from checkpoint.
    Then it loads example image and predicts its label.
    """

    # ckpt can be also a URL!
    CKPT_PATH = "/data1/wangding/project/image-snow-removal/logs/all_UNetResR_bs32_ablation/last.ckpt"

    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it
    trained_model = UNetResRLitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # print model hyperparameters
    print(trained_model.hparams)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()



    
    # load data
    dm = PairedDataModule(
        test_dir='/data1/wangding/datasets/CSD/Test/',
        subdirs=['Snow','Gt','Mask'],
        batch_size=1,
    )
    dm.setup('test')


    trainer = Trainer(gpus='7,')
    trainer.test(model=trained_model, datamodule=dm)


    # img = snow_transforms(img)
    # img = img.reshape((1, *img.size()))  # reshape to form batch of size 1

    # # inference
    # output = trained_model(img)
    # print(output)


if __name__ == "__main__":
    predict()