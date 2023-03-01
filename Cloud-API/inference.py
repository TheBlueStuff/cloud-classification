import torch
import torch.nn.functional as F
import cv2
from utils import *

def infer(model, transform, file):
    orig_img = file.copy()
    file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
    height, width, _ = file.shape
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    model._modules.get('Mixed_7c').register_forward_hook(hook_feature)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    img = transform(image=file)['image']
    img = img.unsqueeze(0)
    cloud_type = 0
    prob = 0
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1).data.squeeze()
        class_idx = torch.topk(probs, 1)[1].int()
        cloud_type = torch.topk(probs, 1)[1].int().item()
        prob = probs[cloud_type].item()
        prob = "{:.2f}".format(prob)
        print(prob)
    CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
    image_base64 = show_cam(CAMs, width, height, orig_img)
    return cloud_type, image_base64, prob


def infer_multiple(model, transform, file):
    img = transform(image=file)['image']
    img = img.unsqueeze(0)
    cloud_type = 0
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        cloud_type = torch.topk(probs, 1)[1].int().item()
    return cloud_type