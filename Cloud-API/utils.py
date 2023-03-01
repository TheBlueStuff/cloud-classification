import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
def show_cam(CAMs, width, height, orig_image):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + orig_image * 0.5
        #Show image if necessary
        #cv2.imshow('CAM', result/255.)
        #cv2.waitKey(0)
        retval, buffer = cv2.imencode('.png', result)
        image_data = io.BytesIO(buffer)
        base64_bytes = base64.b64encode(image_data.getvalue())
        base64_string = base64_bytes.decode('utf-8')
    return base64_string