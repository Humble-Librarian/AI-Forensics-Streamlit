<p align="center">
  <img src="https://img.shields.io/badge/AI%20Forensic%20Console-Media%20Authentication-6366F1?style=for-the-badge&logo=shield&logoColor=white" alt="AI Forensic Console" />
</p>

<h1 align="center">🛡️ AI Forensic Console</h1>
<h3 align="center">Advanced Deepfake Detection & Media Authentication Platform</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Computer%20Vision-OpenCV%20%7C%20MTCNN-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Reporting-ReportLab-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<p align="center">
  <b>Uncover the Truth in Digital Media.</b><br/>
  Upload a video → Get instant, multi-model AI forensic diagnostics evaluating spatial artifacts, frequency noise, and temporal consistency.
</p>

---

## 📸 Screenshots

| Dashboard Interface | Frame Analysis | Forensic Metrics |
|:---:|:---:|:---:|
| System monitoring + drag & drop | Frame-by-frame dot visualization | Authentic/Suspicious/Manipulated gauges |

| Case History | Settings & Config | Exported Reports |
|:---:|:---:|:---:|
| Persistent session-based history logs | Hardware acceleration & model status | Institutional-grade PDF summaries |

---

## ✨ Features

### 🎯 Core Inference Engine
- **Multi-Modal Detection Pipeline** — Synergizes three distinct deep learning architectures to detect manipulations:
  - **Spatial Analysis (SpatialXception)**: Detects visual artifacts and blending boundaries in face crops.
  - **Frequency Analysis (SRMXception)**: Analyzes invisible high-frequency noise patterns and compression artifacts.
  - **Temporal Consistency (DeepfakeLSTM)**: Tracks inter-frame anomalies across sequences.
- **Dynamic Sequence Resolution** — Analyzes a customizable sequence length (10 to 25 frames) for optimized speed vs. accuracy.
- **Explainable Scoring** — Breaks down the final probability score into dedicated spatial, frequency, and temporal probability metrics.

### 🔬 Deep Analysis
- **Frame-by-Frame Tracking** — Visualizes the "fakeness" of individual parsed frames via a color-coded block and dot UI.
- **MTCNN Face Extraction** — Robust facial tracking and cropping powered by `facenet-pytorch`, ensuring models only analyze the ROI (Region of Interest).
- **Automated Metadata Parsing** — Instantly reads and displays video resolution, actual file duration, codec, and framerate.

### 🚀 Reporting & Operations
- **System Kernel Logging** — Real-time event tracking in the UI showing backend statuses, Tensor Core initialization, and inference diagnostics.
- **Hardware Acceleration Check** — Automatically defaults and reports on CUDA/GPU availability for expedited model processing.
- **Institutional PDF Export** — Instantly generates and downloads a structured PDF dossier of the analysis using `ReportLab`, suitable for case files.

### 🖥️ Premium Dashboard
- **Glassmorphism UI** — Modern, cyber-tactical dark mode interface featuring interactive metric rings, pills, and dynamic status badges.
- **Live Memory Management** — Forces manual garbage collection (`gc`) throughout the pipeline to prevent memory leaks during extended usage on large videos.
- **Responsive Layout** — Multi-panel grid design engineered in Custom CSS & Streamlit targeting 4K down to standard 1080p monitor displays.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                    │
│                 (app.py — Port 8501)                    │
└─────────────────────┬───────────────────────────────────┘
                      │ Local Processing
┌─────────────────────▼───────────────────────────────────┐
│                   Inference Pipeline                    │
│               (engine.py / utils.py)                    │
├─────────────────────────────────────────────────────────┤
│  Actions:  Video Read · MTCNN Face Crop · Evaluation    │
├─────────────────────────────────────────────────────────┤
│              Model Orchestrator (PyTorch)               │
│        Connects Neural Networks & Aggregates Output     │
├─────────────────────────────────────────────────────────┤
│                    Neural Layer                         │
│  ┌─────────────────────────┐ ┌────────────────────────┐ │
│  │    SpatialXception      │ │     SRMXception        │ │
│  │   (Pixel Artifacts)     │ │  (High-Freq Noise)     │ │
│  └─────────────────────────┘ └────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │                    DeepfakeLSTM                    │ │
│  │               (Temporal Consistency)               │ │
│  └────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                     Core Services                       │
│  ┌──────────────────────┐  ┌─────────────────────────┐  │
│  │ Metadata Extraction  │  │ PDF Report Generator    │  │
│  │      (OpenCV)        │  │      (ReportLab)        │  │
│  └──────────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit, Plotly, HTML/CSS injected |
| **Backend/Engine** | Python 3.10+, PyTorch (`cuda:0` enabled) |
| **Computer Vision** | OpenCV (`opencv-python-headless`), Pillow |
| **Neural Models** | `facenet-pytorch` (MTCNN), `timm` (Xception backbones) |
| **Reporting** | ReportLab (PDF Synthesis) |
| **Data Tools** | Pandas, NumPy |
| **Cloud Models** | `gdown` (Drive linking for weights caching) |

---

## 📁 Project Structure

```
AI-Forensics-Streamlit/
├── app.py                    # Streamlit frontend dashboard & UI core
├── config.py                 # Application settings, hyperparams & constants
├── engine.py                 # Neural inference logic & frame evaluation
├── models.py                 # PyTorch Spatial, SRM, and LSTM architectures
├── utils.py                  # Extractor funcs, metadata parsing, PDF generation
├── requirements.txt          # Python runtime dependencies
├── styles.css                # Custom CSS for glassmorphism/UI styling
│
├── processed_faces/          # Face crop output directory (auto-created)
├── video_features/           # Extracted features directory (auto-created)
│
└── __pycache__/              # Python bytecode cache
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- **Git**
- **NVIDIA GPU** *(Highly recommended for rapid inference times, though pure CPU works)*

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/AI-Forensics-Streamlit.git
cd AI-Forensics-Streamlit

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
# NOTE: To enable heavy GPU operations, ensure you install the CUDA-flavoured version of PyTorch!
pip install -r requirements.txt
```

### Running the Application

```bash
# Launch the Streamlit Dashboard
streamlit run app.py
```

### Access the Console
Navigate to `http://localhost:8501` in your browser.

*(On initialization, the app will ping cloud instances globally to pull the latest `.pth` tensor weights locally. This happens once.)*

---

## 📖 How It Works

```
Video Upload (.mp4/.mov/.avi)
    │
    ▼
Metadata Extraction ──► Local Temp Caching
    │
    ▼
MTCNN Frame-by-Frame Face Crop (Extracting ROIs)
    │
    ▼
Spatial Analysis                    Frequency Analysis
(Xception Core)                     (SRM Filtered Xception)
    │                                        │
    ▼                                        ▼
    └──────────────► LSTM Layer ◄────────────┘
               (Temporal Sequencing)
                         │
                         ▼
        Fusion Averaging (Weighted Metrics)
                         │
    ┌────────────────────┴────────────────────┐
    │                                         │
    ▼                                         ▼
UI Results Render                     PDF Case Report
(Authentic / Suspect)                (Available for Download)
```

---

## 🤖 Neural Models Used

| Architecture | Purpose | Pre-training / Framework |
|-------|---------|-----------|
| **MTCNN** | Facial extraction and bounding boxes | `facenet-pytorch` |
| **SpatialXception** | Identifying visual blending boundaries | `timm`, ImageNet weights |
| **SRMXception** | Spatial Rich Model filters for invisible noise | `timm`, ImageNet weights |
| **DeepfakeLSTM** | Recurrent layers for inter-frame stability | PyTorch Native |

> System downloads three tailored weights models (`spatial_model.pth`, `srm_model.pth`, `lstm_model.pth`) via `config.py` Google Drive links if they do not exist locally on boot.

---

## 📊 Training Datasets

The core models of the AI Forensic Console were trained and validated across standard forensic benchmark datasets indicating robust generalization qualities:

| Dataset | Type | Sequences Overview |
|---------|---------|---------|
| `FaceForensics++ (FF++)` | Authentic/Real | Original Youtube Actors |
| `FaceForensics++ (FF++)` | Deepfakes | Autoencoder manipulation |
| `FaceForensics++ (FF++)` | Face2Face | Computer Graphics transfers |
| `FaceForensics++ (FF++)` | FaceSwap | Graphics-based swaps |
| `FaceForensics++ (FF++)` | NeuralTextures | GAN-based manipulation |
| `Celeb-DF-v2` | Authentic | High-res Youtube Real sequences |

*(Fake data sourced entirely from FF++ variations due to high-quality artifact density.)*

---

## ⚙️ Configuration

Key environment variables in `config.py`:

```python
VERSION = "3.2.0"          # Application Semantic Versioning
IMG_SIZE = 299             # Target dimensional scaling for neural pass
SEQ_LENGTH = 10            # Default frames isolated per video (10-25 editable in UI)
DEVICE = torch.device      # 'cuda:0' or 'cpu' auto-detected
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/forensic-enhancement`)
3. Commit your changes (`git commit -m 'Added parallel frame extraction'`)
4. Push to the branch (`git push origin feature/forensic-enhancement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using PyTorch, Streamlit & Custom Transformers<br/>
  <b>AI Forensic Console</b> — Uncover the Truth in Digital Media.
</p>