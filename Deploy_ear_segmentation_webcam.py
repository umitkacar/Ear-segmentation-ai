import os
import cv2
import torch
import segmentation_models_pytorch as smp
import imgviz
from const import ACTIVATION,CLASSES,DEVICE,ENCODER,ENCODER_WEIGHTS,LOAD_MODEL_DEPLOY_PATH
from preprocessing import get_preprocessing

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"]=""



if __name__ == "__main__":

    return_frame_status = True
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
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), frame_fps, (640,480))

    while(return_frame_status==True):
        
        return_frame_status, camera_frame_bgr = cap.read()
        #frame = cv2.flip(cv2.transpose(frame), flipCode=1)
        
        frame_rgb = cv2.cvtColor(camera_frame_bgr, cv2.COLOR_BGR2RGB)
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
                                    image=frame_rgb, 
                                    label_names=['background','ear'],
                                    font_size=30,
                                    loc="rb",)

            camera_frame_bgr = cv2.cvtColor(labelviz, cv2.COLOR_BGR2RGB)
        
        #cv2.imshow('Ear Mask',pr_mask)
        cv2.imshow('Ear Image',camera_frame_bgr)
        out.write(camera_frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
