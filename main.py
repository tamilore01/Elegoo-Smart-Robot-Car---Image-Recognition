from image_utils import get_image
from network_utils import create_socket, is_socket_connected, safe_send
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from everywhereml.data.collect import MjpegCollector
import numpy as np
import time
import socket
from logging import basicConfig, INFO
import io
from PIL import Image, ImageOps

MODEL_PATH = "../model/keras_model.h5"
LABELS_PATH = "../model/labels.txt"
ESP_CAMERA_IP = 'http://192.168.4.1:81/stream'
CONTROL_HOST = "192.168.4.1"
CONTROL_PORT = 100
CONFIDENCE_THRESHOLD = 0.75
COMMAND_DELAY = 0.3
MAX_RECONNECT_ATTEMPTS = 5
BASE_RECONNECT_DELAY = 1
MIN_HEARTBEAT_INTERVAL = 0.8

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

def load_components():
    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D}
    )
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

def main_control_loop():
    model, class_names = load_components()
    mjpeg_collector = MjpegCollector(address=ESP_CAMERA_IP)
    basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    reconnect_attempts = 0
    last_heartbeat = time.time()
    s = create_socket()

    while reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
        try:
            if not is_socket_connected(s):
                s.close()
                s = create_socket()
                s.connect((CONTROL_HOST, CONTROL_PORT))
                reconnect_attempts = 0
                print("✓ Connection established")

            if time.time() - last_heartbeat > MIN_HEARTBEAT_INTERVAL:
                if not safe_send(s, b"{Heartbeat}"):
                    raise ConnectionError("Heartbeat failed")
                last_heartbeat = time.time()

            image_data = get_image(mjpeg_collector)
            if image_data is None:
                continue

            prediction = model.predict(image_data, verbose=0)
            index = np.argmax(prediction)
            confidence = prediction[0][index]

            if confidence < CONFIDENCE_THRESHOLD:
                print(f"! Low confidence ({confidence:.2f}) - ignoring prediction")
                continue

            class_name = class_names[index].lower()
            if "people" in class_name:
                print("→ PERSON detected - STOPPING")
                safe_send(s, b'{"N":100}')
            elif "allow" in class_name:
                print("→ ALLOWED object - MOVING")
                time.sleep(1)
                # safe_send(s, b'{"N":3,"D1":3,"D2":100}')
                # time.sleep(1)
                # safe_send(s, b'{"N":100}')

            time.sleep(COMMAND_DELAY)

        except ConnectionError as e:
            reconnect_attempts += 1
            delay = min(BASE_RECONNECT_DELAY * (2 ** reconnect_attempts), 10)
            print(f"! Connection error: {e} - Retrying in {delay}s...")
            time.sleep(delay)
            s = create_socket()

        except KeyboardInterrupt:
            print("
! Controlled shutdown initiated")
            safe_send(s, b'{"N":100}')
            break

        except Exception as e:
            print(f"! Unexpected error: {str(e)}")
            safe_send(s, b'{"N":100}')
            time.sleep(1)

    if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
        print("! Maximum reconnection attempts reached - shutting down")

if __name__ == "__main__":
    print("🚀 Elegoo Smart Car Controller v2.0")
    try:
        test_img = get_image(MjpegCollector(address=ESP_CAMERA_IP))
        if test_img is not None:
            print("✓ All systems nominal - starting control loop")
            main_control_loop()
        else:
            print("! Camera initialization failed")
    except Exception as e:
        print(f"! Fatal startup error: {str(e)}")
    finally:
        print("🔴 System shutdown complete")
