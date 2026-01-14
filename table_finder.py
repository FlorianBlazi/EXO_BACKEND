from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests
import matplotlib.pyplot as plt

class TableFinder:
    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained("TahaDouaji/detr-doc-table-detection")
        self.model = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection")

    def find_tables(self, image):
        """
        finds the tables on invoice and bank documents images

        inuput : image (PIL)

        output : results = {'scores : [for scores >threshold],
                            'labels' : [labels of different tables],
                            'boxes' : [coordinates of tables(top left/bottom right)]}
        """
        image = image.convert("RGB")

        input = self.processor(images=image, return_tensors="pt", do_pad=True)
        output = self.model(**input)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        result = self.processor.post_process_object_detection(output, target_sizes=target_sizes, threshold=0.9)[0]

        if len(result["scores"]) == 0:
            print(result)
            return result
        
        else :
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                print(
                    f"Detected {self.model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )
            return result
    
    def identify_tables(self, image,results):

        """
        boxes the tables in the image in red

        inputs :    -base image (PIL)
                    -results (output of the table detection)
        
        output :    -image with boxed tables (PIL)
        """

        image = image.copy()
        draw = ImageDraw.Draw(image)

        for box in results["boxes"]:
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="red")

        return image