from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os

# Absolute paths to model and image
model_path = r"C:\Marine_detection\best.pt"
image_path = r"C:\Marine_detection\IMG_0034_JPG.rf.226b3980b1eee20762685fce1ba0e560.jpg"

# Load the trained YOLO model
model = YOLO(model_path)

# Run detection on the image
results = model(image_path, save=True, imgsz=640, conf=0.25)

# Get the output directory where YOLO saves predictions
save_dir = results[0].save_dir  # e.g., 'runs/detect/predict', 'runs/detect/predict2', etc.

# Construct path to the saved prediction image
predicted_image_path = os.path.join(save_dir, os.path.basename(image_path))

# Display the result
img = Image.open(predicted_image_path)
plt.imshow(img)
plt.axis('off')
plt.title("Detected Marine Debris")
plt.show()
