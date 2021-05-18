import glob
import os
import cv2
import torch
from albumentations import Compose, Resize,Lambda
import segmentation_models_pytorch as smp

os.environ["CUDA_VISIBLE_DEVICES"]=""

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOAD_MODEL_DEPLOY_PATH = "./model_ear/best_model_ear_v1_43.pth"
ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['optic_disc']
ACTIVATION = 'sigmoid'
DEVICE = "cpu"

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
          Resize(height=320, width=480, always_apply=True),
    ]
    return Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        Lambda(image=preprocessing_fn),
        Lambda(image=to_tensor),
    ]
    return Compose(_transform)

if __name__ == "__main__":

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    model = torch.load(LOAD_MODEL_DEPLOY_PATH, map_location=DEVICE)
    model.eval()
    model.to(DEVICE)
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing = get_preprocessing(preprocessing_fn)

    data_dir =  ["./test-images"]

    data_samples = []
    for _dir in data_dir:
        # JPEG
        _list_tif = glob.glob(_dir + '/*.jpg')
        data_samples += _list_tif

    for path in data_samples:

        img = cv2.imread(path)
        img = cv2.resize(img, (480,320))
        h, w = img.shape[:2]
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            
            #tensor_img = my_transforms(image=image)['image'].unsqueeze(0)
            #predictions = model.forward(tensor_img.to(DEVICE))
            
            sample = preprocessing(image=image)
            image = sample['image']

            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            
            pr_mask = model.predict(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
        cv2.imshow('Ear Mask',pr_mask)
        cv2.imshow('Ear Image',img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
