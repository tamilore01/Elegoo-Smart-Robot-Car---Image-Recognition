import io
import numpy as np
import time
from PIL import Image, ImageOps

IMAGE_SIZE = (224, 224)

def get_image(mjpeg_collector):
    for attempt in range(3):
        try:
            test_sample = mjpeg_collector.collect_by_samples(num_samples=1)
            image = Image.open(io.BytesIO(test_sample[0])).convert("RGB")
            image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
            image_array = np.asarray(image, dtype=np.float32)
            return np.expand_dims((image_array / 127.5) - 1, axis=0)
        except Exception as e:
            print(f"! Image capture failed (attempt {attempt + 1}): {str(e)}")
            time.sleep(0.5)
    return None
