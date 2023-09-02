import os
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from PIL import Image, ExifTags
import json
from io import BytesIO


def init_context(context):
    MODEL_PATH = os.environ["MODEL_PATH"]
    context.model = YOLO(MODEL_PATH)


def handler(context, event):
    try:
        image_data = base64.b64decode(event.body.decode('utf-8'))
        original_image = Image.open(BytesIO(image_data))

        try:
            exif_data = original_image._getexif()
            if exif_data is not None:
                orientation_tag = 274
                if orientation_tag in exif_data:
                    orientation = exif_data[orientation_tag]
                    if orientation == 3:
                        original_image = original_image.rotate(
                            180, expand=True)
                    elif orientation == 6:
                        original_image = original_image.rotate(
                            -90, expand=True)
                    elif orientation == 8:
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

        def closest_point(point, points):
            return min(points, key=lambda p: abs(p[0] - point[0]) + abs(p[1] - point[1]))

        closest_points = [closest_point((0, 0), points),
                          closest_point((width, 0), points),
                          closest_point((0, height), points),
                          closest_point((width, height), points)]

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
