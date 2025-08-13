import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


def test_on_image(image_path, model_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if image exists
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Image loaded successfully: {image_path}")
        print(f"Image size: {image.size}")
    except Exception as e:
        print(f"Error opening image: {e}")
        print(f"Check if the image path is correct: {image_path}")
        return None

    # Check if model exists and load it
    try:
        # Load TorchScript model directly
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print(f"TorchScript model loaded successfully: {model_path}")
    except Exception as e:
        print(f"Error loading TorchScript model: {e}")
        print(f"Check if the model path is correct: {model_path}")
        return None

    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # Match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")

    # Make prediction
    with torch.no_grad():
        try:
            output = model(input_tensor)
            # Handle different output formats
            if isinstance(output, dict) and 'out' in output:
                output = output['out']
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Error during inference: {e}")
            return None

    # Get segmentation mask
    probs = torch.softmax(output, dim=1)
    pred = torch.argmax(probs, dim=1).cpu().numpy()[0]
    print(f"Prediction shape: {pred.shape}")
    print(f"Unique values in prediction: {np.unique(pred)}")

    # Display results
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap='gray')
    plt.title("Segmentation Mask")
    plt.axis('off')

    # Create overlay
    image_np = np.array(image.resize((384, 384)))
    mask_colored = np.zeros_like(image_np)
    
    # Assuming class 1 is debris (you may need to adjust this based on your model)
    mask_colored[pred == 1] = [0, 255, 0]  # Green for debris
    
    overlay = cv2.addWeighted(image_np, 0.7, mask_colored, 0.3, 0)

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('segmentation_results.png')
    print("Results saved to segmentation_results.png")
    plt.show()

    # Calculate debris percentage
    debris_percent = (np.sum(pred == 1) / pred.size) * 100
    print(f"Debris coverage: {debris_percent:.2f}%")

    return pred


# Execute the function with your paths
from pathlib import Path

if __name__ == "__main__":
    result = test_on_image(
        image_path=Path(r"C:\Users\RIFATH\Downloads\marine-debris-problem.jpg"),
        model_path=Path(r"C:\Marine_detection\deeplabv3plus_marine_debris_best (1).pt")
    )