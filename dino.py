from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
import torch
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
# Load the model

text = "rocket"

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


# Predict a segment based on a point prompt
image_path = "/home/ericp/autodistill/frames/Amateur rocket reaches 121,000 ft [sQw_C5KLhFM].webm_230.png"
# image_path = "/home/ericp/autodistill/frames/Amateur rocket reaches 121,000 ft [sQw_C5KLhFM].webm_150.png"
image = Image.open(image_path)
inputs = processor(images=image, text=preprocess_caption(text), return_tensors="pt")
print("predicting")
with torch.no_grad():
  outputs = model(**inputs)

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        label = f'{text}: {score:0.2f}'
        ax.text(xmin, ymin, label, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig("rocket_predicted.png")
width, height = image.size
postprocessed_outputs = processor.image_processor.post_process_object_detection(outputs,
                                                                target_sizes=[(height, width)],
                                                                threshold=0.3)
results = postprocessed_outputs[0]
plot_results(image, results['scores'].tolist(), results['labels'].tolist(), results['boxes'].tolist())
