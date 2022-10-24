import torch
import torch.nn.functional as F

def infer(model, transform, file):
    img = transform(image=file)['image']
    img = img.unsqueeze(0)
    cloud_type = 0
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        cloud_type = torch.topk(probs, 1)[1].int().item()
    return cloud_type, file


def infer_multiple(model, transform, file):
    img = transform(image=file)['image']
    img = img.unsqueeze(0)
    cloud_type = 0
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        cloud_type = cloud_type = torch.topk(probs, 1)[1].int().item()
    return cloud_type