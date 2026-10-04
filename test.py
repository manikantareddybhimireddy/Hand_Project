import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSD
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


# Load Hand X-ray Classification Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hand_xray_model = models.resnet50(weights=None)
hand_xray_model.fc = torch.nn.Linear(hand_xray_model.fc.in_features, 2)
hand_xray_model.load_state_dict(torch.load("model\hand_xray_model.pth", map_location=device))
hand_xray_model.to(device)
hand_xray_model.eval()

# Image transformation for classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def is_hand_xray(image):
    """True if the image is a Hand X-ray, otherwise False"""
    image = Image.open(image).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = hand_xray_model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item() == 0  # 0 → Hand X-ray, 1 → Other


classes=['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
num_classes = 7

def get_model():
  
    model=torchvision.models.detection.fasterrcnn_resnet50_fpn(preTrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=7)
    model.load_state_dict(torch.load("model\Resnet.pt", map_location='cpu'))
    
    return model


def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : 
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]
    
    return preds


def plot_image_from_output(img, annotation):
    
    img = img.cpu().detach().permute(1, 2, 0).numpy()    
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis('off')
    
    class_name = None
    
    if annotation and "scores" in annotation and len(annotation["scores"]) > 0:
        max_score_idx = torch.argmax(annotation["scores"][0])

        # Extract the coordinates of the bounding box with the highest score
        xmin, ymin, xmax, ymax = annotation["boxes"][max_score_idx].detach().cpu().numpy()
        label_idx = annotation["labels"][max_score_idx]
        
        class_name = classes[label_idx]

        # Plot the bounding box with the highest score
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # ax.text(xmin, ymin - 10, class_name, fontsize=12, color='orange', fontweight='bold')

    return fig, ax, class_name

    
def figure_to_array(fig):

    fig.canvas.draw()
    
    return np.array(fig.canvas.renderer._renderer)