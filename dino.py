from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor, GroundingDinoImageProcessor
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
import os
import yaml

CURRENT_ABSPATH = os.path.dirname(os.path.realpath(__file__))
out_path = "yolo_output"
split_proportions = [0.8, 0.1, 0.1]
split_names = ["train", "valid", "test"]
os.makedirs(out_path, exist_ok=True)
for split_name in split_names:
    os.makedirs(f"{out_path}/{split_name}", exist_ok=True)
    os.makedirs(f"{out_path}/{split_name}/images", exist_ok=True)
    os.makedirs(f"{out_path}/{split_name}/labels", exist_ok=True)

metadata = {
    "path": f"{CURRENT_ABSPATH}/{out_path}",
    "train": f"{CURRENT_ABSPATH}/{out_path}/train/images",
    "val": f"{CURRENT_ABSPATH}/{out_path}/valid/images",
    "test": f"{CURRENT_ABSPATH}/{out_path}/test/images",
    "nc": 1,
    "names": ["rocket"]
}
yaml.dump(metadata, open(f"{out_path}/metadata.yaml", "w"))

processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to('cuda')
prompt_text = "rocket." # needs to be lowercase and end with a period

for frame_fname in tqdm(os.listdir("frames")):
    frame_stem = frame_fname.split(".")[0]
    image = Image.open(f"frames/{frame_fname}")
    inputs = processor(images=image, text=prompt_text, return_tensors="pt")
    for k,v in inputs.items():
        if type(v) == torch.Tensor:
            inputs[k] = v.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    width, height = image.size
    postprocessed_outputs = processor.image_processor.post_process_object_detection(outputs,
                                                                target_sizes=[(height, width)],
                                                                threshold=0.3)
    results = postprocessed_outputs[0] 

    scores, labels, boxes = results['scores'].tolist(), results['labels'].tolist(), results['boxes'].tolist()
    if len(scores) == 0 and np.random.rand() < 0.9:
        continue
    spit_name = np.random.choice(split_names, p=split_proportions)
    image.save(f"{out_path}/{split_name}/images/{frame_stem}.jpg")
    with open(f"{out_path}/{split_name}/labels/{frame_stem}.txt", "w") as f:
        for score, label, (xmin, ymin, xmax, ymax) in zip(scores, labels, boxes):
            x_float = (xmin + xmax) / 2 / width
            y_float = (ymin + ymax) / 2 / height
            width_float = (xmax - xmin) / width
            height_float = (ymax - ymin) / height
            f.write(f"0 {x_float} {y_float} {width_float} {height_float}\n")
