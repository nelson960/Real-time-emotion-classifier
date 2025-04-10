
# 😃 Facial Emotion Recognition with MobileViTv2

A real-time **Facial Emotion Recognition (FER)** system built with **MobileViTv2**, fine-tuned on the **FANE dataset**, and optimized for deployment on **constrained hardware** like the **MacBook Pro M2 Pro (16GB RAM)**.

---
## 🌟 Highlights

✅ Fine-tuned `MobileViTv2` for robust emotion recognition  
✅ Real-time inference with **efficient resource usage**  
✅ Designed for **edge deployment** on MacBooks and similar low-power devices  
✅ Achieved **64.53% accuracy** and **0.6415 F1 score** on validation — **on just a MacBook Pro M2 Pro with 16GB RAM**  
✅ Containerized with **Docker + Streamlit** for cross-platform deployment  
✅ Webcam-ready using either native video device or browser-based Streamlit WebRTC

---
## 🧠 Model Training Description

This project leverages transfer learning and a progressive fine-tuning approach with **MobileViTv2** to recognize emotions from facial expressions. Below is a detailed description of the training pipeline:

### 📁 Dataset: FANE (Facial Emotion Dataset in the Wild)

- Split into:
  - **Train**: 70%
  - **Validation**: 15%
  - **Test**: 15%
- Verified for **class distribution consistency** across splits.
- Contains images labeled with **9 emotion classes**.

---
### ⚙️ Improved Training Strategy

The model uses **transfer learning** with advanced fine-tuning optimizations to deliver strong results even on constrained systems.

#### 🔄 Gradual Layer Unfreezing

| Stage | Epoch | Layers Unfrozen |
|-------|-------|------------------|
| 4     | 0     | Classifier only  |
| 3     | 1     | Deeper Blocks    |
| 2     | 3     | Mid-Level Layers |
| 1     | 5     | Full Model       |

This progressive approach avoids overfitting and **preserves pretrained features** from ImageNet.

---

### 🛠️ Advanced Techniques Used

| Technique                    | Purpose                                                                 |
|-----------------------------|-------------------------------------------------------------------------|
| ✅ Mixup Augmentation        | Regularization and better generalization                               |
| ✅ Class-Balanced Focal Loss | Emphasize underrepresented emotions                                    |
| ✅ Gradient Accumulation     | Simulated large batch size under 16GB RAM                              |
| ✅ Gradient Clipping         | Prevent exploding gradients                                             |
| ✅ EMA (Exponential Moving Avg) | Stabilize weight updates for better validation metrics            |
| ✅ OneCycleLR Scheduler      | Dynamic learning rate for faster convergence                           |
| ✅ Temperature Scaling       | Calibrate prediction confidence for post-training reliability           |

---

### 📈 Results on MacBook Pro M2 Pro (16GB RAM)

- **Best Checkpoint Epoch**: 47  
- **Validation Accuracy**: `64.53%`  
- **Weighted F1 Score**: `0.6415`  
- **Validation Loss**: `0.6118`  
- ✅ Achieved with **batch size optimization**, **gradient accumulation**, and **progressive unfreezing** under MacBook hardware constraints

---
## Running via Streamlit
```bash
OPENCV_AVFOUNDATION_SKIP_AUTH=1 streamlit run streamlit/app.py
```

## Running via FastAPI
```bash
uvicorn main:app --reload

curl -X POST http://127.0.0.1:8000/predict \
     -F "file=@/path/to/your/image.jpg"

```
## 🚀 Deploy via Docker

This project is fully containerized. Follow the steps below to build and run the app locally.

### 🛠️ Step 1: Build the Docker Image

From the project root, run:

```bash
docker build -t emotion-streamlit-app .
```

### ▶️ Step 2: Run the Docker Container

```bash 
docker run -p 8501:8501 --name emotion-app emotion-streamlit-app
```
### 🎥 Webcam Access in Docker (Optional but Required for Real-Time Inference)
```bash 
docker run --device=/dev/video0 -p 8501:8501 --name emotion-app emotion-streamlit-app
```