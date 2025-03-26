from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

# 🔥 Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_image_caption(image_path):
    """
    Generates a short caption for the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string representing the caption for the image.
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return f"Error: Unable to load image. {e}"

    model_name = "Salesforce/blip-image-captioning-large"

    # Load model and processor
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    # Process image
    inputs = processor(image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=20)

    # Decode caption
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def detect_objects(image_path):
    """
    Detects objects in the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string with all detected objects.
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return f"Error: Unable to load image. {e}"

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Format detections
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            "label": model.config.id2label[int(label)],
            "confidence": round(float(score), 3)
        })

    return detections if detections else "No objects detected."


if __name__ == '__main__':
    image_path = '"E:\download.jpg"'

    print("🔹 Generating Image Caption...")
    caption = get_image_caption(image_path)
    print(f"📸 Caption: {caption}\n")

    print("🔹 Detecting Objects...")
    detections = detect_objects(image_path)
    print("🛠️ Detections:", detections)
