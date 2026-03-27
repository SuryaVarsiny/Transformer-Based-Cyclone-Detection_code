
# STEP 1: Mount Google Drive

from google.colab import drive
drive.mount('/content/drive')

# STEP 2: Import Libraries

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab import files
import os

# STEP 3: Custom Patch Embedding (Required)

@register_keras_serializable()
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Conv2D(embed_dim,
                                  kernel_size=patch_size,
                                  strides=patch_size)

    def call(self, x):
        x = self.proj(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1, x.shape[-1]])
        return x

# STEP 4: Load DeiT Model

MODEL_PATH = "/content/drive/MyDrive/deit_cyclone_2025.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found. Check path.")

print("Loading DeiT Cyclone Model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"PatchEmbedding": PatchEmbedding},
    compile=False
)
print("✅ Model Loaded Successfully!\n")

IMG_SIZE = 224

# STEP 5: Upload Images (sequence order)

print("Upload satellite image sequence...")
uploaded = files.upload()

# STEP 6: Image Metrics

def image_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    brightness = np.mean(gray)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    texture = np.mean(np.sqrt(sobelx**2 + sobely**2))

    return brightness, texture

# STEP 7: Optical Flow

def optical_flow(prev_img, curr_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    return np.mean(mag)

# STEP 8: Prediction Function

def predict_image(img_np):
    img_norm = img_np / 255.0
    img_input = np.expand_dims(img_norm, axis=0)
    prob = float(model.predict(img_input, verbose=0)[0][0])
    return prob

# STEP 9: Process Images

images = []
filenames = sorted(uploaded.keys())

for file in filenames:
    img = Image.open(file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    images.append(np.array(img))


# STEP 10: Present Frame Detection

print("\n===== PRESENT FRAME DETECTION =====")

present_results = []
prev_img = None
display_images = []

for img, name in zip(images, filenames):

    prob = predict_image(img)
    label = "Cyclone" if prob >= 0.75 else "No Cyclone"

    brightness, texture = image_metrics(img)
    flow = optical_flow(prev_img, img) if prev_img is not None else 0

    present_results.append((brightness, texture, flow))

    # annotate image
    disp = img.copy()
    cv2.putText(disp, f"{label} ({prob:.2f})", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,255,0) if label=="No Cyclone" else (255,0,0), 2)

    display_images.append(disp)
    prev_img = img
# SHOW IMAGES IN ONE HORIZONTAL ROW

plt.figure(figsize=(20,4))

for i, im in enumerate(display_images):
    plt.subplot(1, len(display_images), i+1)
    plt.imshow(im)
    plt.title(filenames[i], fontsize=8)
    plt.axis("off")

plt.suptitle("Cyclone Detection Sequence", fontsize=14)
plt.show()

# STEP 11: Past Time-Series Analysis (t-8 → t-3)

print("\nPAST TIME SERIES ANALYSIS (t-8 → t-1)")

if len(images) >= 8:

    past_frames = images[-8:-3]

    flows = []
    bright_vals = []
    texture_vals = []

    prev = None

    for frame in past_frames:
        b, t = image_metrics(frame)
        bright_vals.append(b)
        texture_vals.append(t)

        if prev is not None:
            flows.append(optical_flow(prev, frame))
        prev = frame

    avg_flow = np.mean(flows)
    flow_trend = flows[-1] - flows[0]

    print(f"Average Flow: {avg_flow:.4f}")
    print(f"Flow Trend: {flow_trend:.4f}")

    if avg_flow > 1.2 and flow_trend > 0:
        past_label = "Cyclone Evolution Detected"
    else:
        past_label = "No Cyclone Evolution"

    print("Past Sequence Result:", past_label)

else:
    print("⚠ Upload at least 8 images for past sequence analysis")

# STEP 12: Plot Temporal Evolution

brightness_series = [x[0] for x in present_results]
texture_series = [x[1] for x in present_results]
flow_series = [x[2] for x in present_results]

plt.figure(figsize=(12,4))
plt.plot(brightness_series, label="Brightness")
plt.plot(texture_series, label="Texture")
plt.plot(flow_series, label="Optical Flow")
plt.title("Spatiotemporal Evolution")
plt.xlabel("Frame Index")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()
