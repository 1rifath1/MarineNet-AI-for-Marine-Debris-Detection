
# 🌊 Marine Debris Detection & Segmentation

An **AI**-powered application for **real-time marine debris analysis** using **YOLOv8** for object detection and **DeepLabv3+** for semantic segmentation, built with **Streamlit** for an interactive user experience.

# 📌 Overview

Marine plastic pollution is a growing global challenge, threatening marine life, ecosystems, and coastal communities. This project combines state-of-the-art **object detection** and **pixel-level segmentation** techniques to:

* Detect floating or stranded marine **debris**.
* Generate segmentation masks to locate debris precisely.
* Calculate debris coverage percentage for quantitative analysis.

## 🚀 Key Features

* **Dual Model Pipeline**:

  * **YOLOv8** → Object detection for bounding box-level identification.
  * **DeepLabv3+ (ResNet-101)** → Pixel-wise semantic segmentation.
* **Combined Analysis Mode** to visualize both detection and segmentation results side-by-side.
* **Adjustable Detection Confidence Threshold** for fine-tuning performance.
* **Interactive Web Interface** powered by **Streamlit**.
* **Debris Coverage Calculation** for quantitative environmental assessment.
* **Customizable & Extensible** for different datasets and environments.

## 🖥️ Tech Stack

* **Frontend**: Streamlit
* **Backend Models**:

  * YOLOv8 (Ultralytics) – Object detection
  * DeepLabv3+ (ResNet101 backbone) – Semantic segmentation
* **Languages & Libraries**: Python, PyTorch, Torchvision, NumPy, Pillow, Matplotlib

## 📂 Project Structure

```
marine-debris-detection/
├── app.py                                # Streamlit web app  
├── detection_segmentation(combinedmodel).py  # Combined model test script  
├── requirements.txt                      # Dependencies  
├── models/                               
│   ├── best.pt                           # YOLOv8 trained weights  
│   └── deeplabv3plus_marine_debris.pt    # DeepLabv3+ TorchScript model  
└── README.md                             # Documentation  
```

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/marine-debris-detection.git
cd marine-debris-detection

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Running the App

1. Place your trained models in the `models/` directory.
2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open the provided local URL in your browser (usually `http://localhost:8501`).

## 📊 Example Outputs

* **YOLOv8 Detection**: Bounding boxes around marine debris.
* **DeepLabv3+ Segmentation**: Color masks for debris coverage.
* **Combined View**: Side-by-side detection, segmentation, and metrics.

## 📜 License

[MIT License](LICENSE) – You’re free to use, modify, and distribute this work with proper attribution.

## 🙏 Acknowledgments

* **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** for object detection.
* **DeepLabv3+** architecture for semantic segmentation.
* **Streamlit** for interactive UI.


