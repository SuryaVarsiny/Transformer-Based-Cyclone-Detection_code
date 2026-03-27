import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab import files
@register_keras_serializable()
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)

    def call(self, x):
        x = self.proj(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1, x.shape[-1]])
        return x


MODEL_PATH = "/content/drive/MyDrive/deit_cyclone_2025.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"PatchEmbedding": PatchEmbedding}
)

IMG_SIZE = 224
uploaded = files.upload()


def image_intensity_metrics(image, show_gradient=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    texture_energy = np.mean(gradient_magnitude)

    # OPTIONAL visualization
    if show_gradient:
        plt.figure(figsize=(4,4))
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude Map")
        plt.axis('off')
        plt.show()

    return mean_brightness, std_brightness, texture_energy



def compute_optical_flow(prev_img, curr_img, visualize=False):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    mean_flow = np.mean(magnitude)

    if visualize:
        step = 10
        plt.figure(figsize=(4,4))
        plt.imshow(prev_gray, cmap='gray')

        for y in range(0, flow.shape[0], step):
            for x in range(0, flow.shape[1], step):
                fx, fy = flow[y, x]
                plt.arrow(x, y, fx, fy)

        plt.title("Optical Flow Field")
        plt.axis('off')
        plt.show()

    return mean_flow


def annotate_image(image, label, prob, confidence, brightness, texture, flow):
    img = image.copy()
    h, w, _ = img.shape

    if label == "Cyclone":
        color = (255, 0, 0)
    elif label == "No Cyclone":
        color = (0, 255, 0)
    else:
        color = (0, 255, 255)

    cv2.rectangle(img, (w//4, h//4), (3*w//4, 3*h//4), color, 2)

    texts = [
        f"Prediction: {label}",
        f"Model Probability: {prob:.3f}",
        f"Confidence: {confidence*100:.2f}%",
        f"Mean Brightness: {brightness:.1f}",
        f"Texture Energy: {texture:.1f}",
        f"Optical Flow: {flow:.3f}" if flow is not None else "Optical Flow: N/A"
    ]

    y = 15
    for t in texts:
        cv2.putText(img, t, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 18

    return img


CYCLONE_THRESHOLD = 0.75
NO_CYCLONE_THRESHOLD = 0.25

previous_image = None
time_series = []

for filename in sorted(uploaded.keys()):
    image = Image.open(filename).convert("RGB")
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(image_resized)

    img_norm = img_np / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # Model prediction
    prob = float(model.predict(img_input, verbose=0)[0][0])

    if prob >= CYCLONE_THRESHOLD:
        label = "Cyclone"
        confidence = prob
    elif prob <= NO_CYCLONE_THRESHOLD:
        label = "No Cyclone"
        confidence = 1 - prob
    else:
        label = "Uncertain"
        confidence = max(prob, 1 - prob)

    # Image-derived metrics
    brightness, std_brightness, texture = image_intensity_metrics(img_np)

    if previous_image is not None:
        flow = compute_optical_flow(previous_image, img_np)
    else:
        flow = None

    time_series.append({
        "brightness": brightness,
        "texture": texture,
        "flow": flow
    })

    annotated = annotate_image(
        img_np, label, prob, confidence,
        brightness, texture, flow
    )

    previous_image = img_np

    plt.figure(figsize=(5, 5))
    plt.imshow(annotated)
    plt.axis("off")
    plt.title(filename)
    plt.show()


brightness_series = [x["brightness"] for x in time_series]
texture_series = [x["texture"] for x in time_series]
flow_series = [x["flow"] for x in time_series if x["flow"] is not None]

plt.figure(figsize=(12, 4))
plt.plot(brightness_series, label="Mean Brightness")
plt.plot(texture_series, label="Texture Energy")
plt.plot(flow_series, label="Optical Flow")
plt.xlabel("Time (Image Index)")
plt.ylabel("Measured Value")
plt.title("Image-Derived Spatiotemporal Metrics")
plt.legend()
plt.grid(True)
plt.show()
