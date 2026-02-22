# 🛡️ AI Forensic Console (v3.1.0)

A high-fidelity digital forensic application for deepfake detection using multi-modal neural analysis.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://ai-forensics-stream-deployment.streamlit.app/)

## 📽️ Project Overview

The **AI Forensic Console** is a specialized tool designed for investigators and researchers to verify the authenticity of video content. By leveraging a combination of spatial, frequency, and temporal analysis, the system provides a probabilistic score of manipulation with high precision.

### 🧠 Analysis Pipeline
1.  **Face Localization**: MTCNN-based detection to lock onto primary subjects.
2.  **Spatial Analysis**: XceptionNet backbone trained to identify pixel-level GAN artifacts.
3.  **Frequency Analysis**: SRM (Spatial Richness Model) filters used to extract high-frequency noise residuals, followed by Xception-based classification.
4.  **Temporal Consistency**: Bi-Directional LSTM (Long Short-Term Memory) network to detect unnatural transitions and frame-to-frame inconsistencies.
5.  **Multi-Modal Fusion**: Weighted aggregation of scores to produce a final forensic verdict.

---

## 🏗️ Architecture

The codebase is highly modularized for maintainability and scalability:

-   `app.py`: Main Streamlit application entry point (UI & Session Management).
-   `engine.py`: Core inference logic, model loading, and frame extraction.
-   `models.py`: Neural network architecture definitions (Spatial, SRM, LSTM).
-   `utils.py`: Utility functions for video metadata and PDF report generation.
-   `config.py`: Centralized configuration for hyperparameters, Drive IDs, and UI constants.
-   `styles.css`: Custom CSS for the advanced forensic UI theme.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (optional but recommended for faster inference)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AI-Forensics-Streamlit.git
   cd AI-Forensics-Streamlit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Features
-   **Real-time Kernel Log**: Monitor every step of the forensic scan.
-   **Frame-by-Frame Heatmap**: Visualize manipulation probability across the video timeline.
-   **Institutional Reports**: Export detailed PDF forensic reports with case metadata and visual evidence.
-   **Cloud Weights**: Automatic synchronization of latest model weights from secure cloud storage.

---

## ⚖️ Disclaimer
This tool is intended for research and professional forensic assistance. AI-generated results should always be peer-reviewed by qualified human analysts before being used in formal legal proceedings.

---
© 2024 AI Forensic Console Project | [Deployment](https://ai-forensics-stream-deployment.streamlit.app/)