from langchain.tools import BaseTool
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    DetrImageProcessor,
    DetrForObjectDetection,
)
from PIL import Image
import torch
from typing import ClassVar, List


class ImageCaptionTool(BaseTool):
    name: ClassVar[str] = "Image Captioner"
    description: ClassVar[str] = (
        "Use this tool when given the path to an image that you would like to be described. "
        "It will return a simple caption describing the image."
    )

    def _run(self, img_path: str) -> str:
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            return f"Error: Unable to load image. {e}"

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and processor
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        # Process image
        inputs = processor(image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        # Decode caption
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class ObjectDetectionTool(BaseTool):
    name: ClassVar[str] = "Object Detector"
    description: ClassVar[str] = (
        "Use this tool when given the path to an image that you would like to detect objects. "
        "It will return a structured list of detected objects with bounding box coordinates, class names, and confidence scores."
    )

    def _run(self, img_path: str) -> List[dict]:
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            return [{"error": f"Unable to load image. {e}"}]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        # Convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "label": model.config.id2label[int(label)],
                "confidence": round(float(score), 3)
            })

        return detections if detections else [{"message": "No objects detected."}]

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
