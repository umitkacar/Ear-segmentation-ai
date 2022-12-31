from cv2 import Mat
from torch import Tensor, from_numpy


def get_prediction(preprocessing, image: Mat, device: str, model) -> Tensor:
    image = preprocessing(image=image)["image"]
    x_tensor: Tensor = from_numpy(image).to(device).unsqueeze(0)
    predict_mask: Tensor = model.predict(x_tensor)
    return predict_mask.squeeze().cpu().numpy().round()
