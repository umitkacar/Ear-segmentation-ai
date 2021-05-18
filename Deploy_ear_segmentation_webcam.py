import os
import cv2
import torch
from albumentations import Compose, Resize,Lambda
import segmentation_models_pytorch as smp
import imgviz

os.environ["CUDA_VISIBLE_DEVICES"]=""

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOAD_MODEL_DEPLOY_PATH = "./model_ear/best_model_ear_v1_43.pth"
ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['ear']
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

    # Wecam acquisition
    cap = cv2.VideoCapture(0)
    # Webcam resolution
    # cap.set(int(3),640)
    # cap.set(int(4),480)

    # # Video Writer 
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(frame_width)
    print(frame_height)
    frame_fps = 20
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), frame_fps, (640,480))

    num = 0
    while(1):
        ret, frame_bgr = cap.read()
        #frame = cv2.flip(cv2.transpose(frame), flipCode=1)
        if ret==True:
            num = num + 1
        
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb_resize = cv2.resize(frame_rgb, (480,320))

            with torch.no_grad():
                
                #tensor_img = my_transforms(image=image)['image'].unsqueeze(0)
                #predictions = model.forward(tensor_img.to(DEVICE))
                
                sample = preprocessing(image=frame_rgb_resize)
                image = sample['image']

                x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                
                pr_mask = model.predict(x_tensor)
                pr_mask = (pr_mask.squeeze().cpu().numpy().round())

                pr_mask_orj = cv2.resize(pr_mask,(frame_rgb.shape[1],frame_rgb.shape[0]))
    
                # colorize label image
                class_label = pr_mask_orj.squeeze().astype(int)
                labelviz = imgviz.label2rgb(class_label,
                                        img=frame_rgb, 
                                        label_names=['background','ear'],
                                        font_size=30,
                                        loc="rb",)

                frame_bgr = cv2.cvtColor(labelviz, cv2.COLOR_BGR2RGB)
            
            #cv2.imshow('Ear Mask',pr_mask)
            cv2.imshow('Ear Image',frame_bgr)
            out.write(frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
