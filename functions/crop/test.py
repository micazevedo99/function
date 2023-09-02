import os
import cv2
import numpy as np
# Make sure you have this package or replace with your own YOLO implementation
from ultralytics import YOLO
import base64
from PIL import Image, ExifTags
import json
from io import BytesIO


class Context:
    def __init__(self):
        self.model = None

    class Response:
        def __init__(self, body, headers, content_type, status_code):
            self.body = body
            self.headers = headers
            self.content_type = content_type
            self.status_code = status_code


class Event:
    def __init__(self, body):
        self.body = body


def init_context(context):
    MODEL_PATH = "best.pt"  # Replace with your model path
    context.model = YOLO(MODEL_PATH)


def handler(context, event):
    try:
        # Conditional decoding
        if isinstance(event.body, bytes):
            image_data = base64.b64decode(event.body.decode('utf-8'))
        else:
            image_data = base64.b64decode(event.body)

        original_image = Image.open(BytesIO(image_data))

        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = original_image._getexif()
            if exif is not None and orientation in exif:
                if exif[orientation] == 3:
                    original_image = original_image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    original_image = original_image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    original_image = original_image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass

        original_image = original_image.convert("RGB")
        image_np = np.array(original_image)
        original_image_np = np.copy(image_np)
        height, width, _ = image_np.shape

        results = context.model.predict(source=[image_np], save=False)
        masks = results[0].masks
        points = masks.segments[0]
        points = (np.array(points, dtype=np.float32).reshape(
            (-1, 2)) * [width, height]).astype(int)

        target_aspect_ratio = 0.73
        target_width = 600
        target_height = int(target_width / target_aspect_ratio)

        closest_points = [min(points, key=lambda p: abs(p[0] - point[0]) + abs(p[1] - point[1])) for point in [
            (0, 0), (width, 0), (0, height), (width, height)]]

        pts1 = np.float32(closest_points)
        pts2 = np.float32([(0, 0), (target_width, 0),
                           (0, target_height), (target_width, target_height)])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(
            original_image_np, M, (target_width, target_height))

        buffered = BytesIO()
        image_pil = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        image_pil.save(buffered, format="JPEG")
        cropped_base64 = base64.b64encode(buffered.getvalue()).decode()

        return context.Response(body=json.dumps({"image": cropped_base64}),
                                headers={},
                                content_type='application/json',
                                status_code=200)

    except Exception as e:
        return context.Response(body=json.dumps({"error": str(e)}),
                                headers={},
                                content_type='application/json',
                                status_code=500)


def main():
    context = Context()
    init_context(context)

    # Load an image and convert it to base64 for testing
    with open("/Users/mike/Desktop/server/static/uploaded/IMG_2786.jpeg", "rb") as f:
        test_image_base64 = base64.b64encode(f.read()).decode("utf-8")

    event = Event(test_image_base64)
    response = handler(context, event)

    print(response.body)
    print(response.status_code)


if __name__ == "__main__":
    main()
