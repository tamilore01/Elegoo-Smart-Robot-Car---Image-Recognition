import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable OneDNN optimizations for compatibility

import io
import time
import socket
import numpy as np
from logging import basicConfig, INFO
from PIL import Image, ImageOps
from everywhereml.data.collect import MjpegCollector
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# =============================================
# CUSTOM LAYER FIX FOR COMPATIBILITY
# =============================================
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)  # Remove problematic argument
        super().__init__(*args, **kwargs)

# =============================================
# CONFIGURATION
# =============================================
MODEL_PATH = "C:/Users/kemcg/Downloads/converted_keras(1)/keras_model.h5"
LABELS_PATH = "C:/Users/kemcg/Downloads/converted_keras(1)/labels.txt"
ESP_CAMERA_IP = 'http://192.168.4.1:81/stream'
CONTROL_HOST = "192.168.4.1"
CONTROL_PORT = 100
HEARTBEAT_INTERVAL = 2  # Seconds between heartbeats
IMAGE_SIZE = (224, 224)

# =============================================
# INITIALIZATION
# =============================================
# Load model with custom layer fix
try:
    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D}
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    exit(1)

# Load labels
try:
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print("Labels loaded successfully")
except Exception as e:
    print(f"Failed to load labels: {str(e)}")
    exit(1)

# Initialize camera collector
basicConfig(level=INFO)
mjpeg_collector = MjpegCollector(address=ESP_CAMERA_IP)

# =============================================
# IMAGE PROCESSING FUNCTIONS
# =============================================
def get_image(mjpeg_collector):
    """Capture and preprocess image from camera"""
    try:
        # Capture image
        test_sample = mjpeg_collector.collect_by_samples(num_samples=1)
        
        # Convert to RGB and process
        image = Image.open(io.BytesIO(test_sample[0])).convert("RGB")
        image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.asarray(image, dtype=np.float32)
        normalized_image_array = (image_array / 127.5) - 1
        
        # Expand dimensions for model input
        return np.expand_dims(normalized_image_array, axis=0)
        
    except Exception as e:
        print(f"Error capturing image: {str(e)}")
        return None

# =============================================
# MAIN CONTROL LOOP
# =============================================
def main_control_loop():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((CONTROL_HOST, CONTROL_PORT))
            print(f"Connected to {CONTROL_HOST}:{CONTROL_PORT}")
            
            last_heartbeat = time.time()
            
            while True:
                # Send heartbeat if needed
                if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                    s.sendall(b"{Heartbeat}")
                    last_heartbeat = time.time()
                    print("Sent heartbeat")
                
                # Capture and process image
                print("Capturing image...")
                image_data = get_image(mjpeg_collector)
                if image_data is None:
                    continue
                
                # Make prediction
                prediction = model.predict(image_data)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence = prediction[0][index]
                
                print(f"Predicted: {class_name} (Confidence: {confidence:.2f})")
                
                # Control logic
                if "people" in class_name.lower() and confidence > 0.7:  # Confidence threshold
                    print("Person detected - taking action")
                    s.sendall(b'{"N":100}')  # Stop command
                    time.sleep(1)
                elif "allow" in class_name.lower() and confidence > 0.7:
                    print("Allowed object - moving forward")
                    s.sendall(b'{"N":102,"D1":1,"D2":20}')  # Move forward
                    time.sleep(1)
                    s.sendall(b'{"N":100}')  # Stop after 1 second
                
                time.sleep(0.5)  # Small delay between cycles
                
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"Error in control loop: {str(e)}")
        finally:
            s.sendall(b'{"N":100}')  # Ensure car stops when exiting
            print("Connection closed")

# =============================================
# MAIN EXECUTION
# =============================================
if __name__ == "__main__":
    print("Starting Smart Car Control System")
    try:
        # Initial test capture
        test_data = get_image(mjpeg_collector)
        if test_data is not None:
            print("Camera test successful")
            main_control_loop()
        else:
            print("Failed initial camera test")
    except Exception as e:
        print(f"Fatal error: {str(e)}")