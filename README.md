
# 😃 Facial Emotion Recognition with MobileViTv2

This project delivers a real-time **Facial Emotion Recognition (FER)** system using a fine-tuned **MobileViTv2** model on the **FANE dataset**. The model predicts 9 different emotions from facial images and runs seamlessly in a **Dockerized Streamlit application**.

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

### 🔧 Training Workflow

#### 1. **Pretrained Model Loading**
- Load `mobilevitv2_100` pretrained on ImageNet.

#### 2. **Initial Freezing**
- Freeze all backbone layers to preserve learned representations.

#### 3. **Gradual Unfreezing**
- At specific epochs, deeper layers are unfrozen to allow better task adaptation:
  - **Stage 4**: Unfrozen at Epoch 0
  - **Stage 3**: Epoch 1
  - **Stage 2**: Epoch 3
  - **Stage 1**: Epoch 5

This strategy prevents catastrophic forgetting while fine-tuning.

#### 4. **Advanced Training Techniques**

- **Mixup Augmentation**: Soft-label image blending to improve generalization.
- **Class-Balanced Focal Loss**: Focus on hard examples and manage emotion class imbalance.
- **Gradient Accumulation**: Simulates larger batch size under memory constraints.
- **Gradient Clipping**: Prevents exploding gradients in deep layers.
- **Exponential Moving Average (EMA)**: Stabilizes training by averaging model weights.
- **OneCycleLR**: Dynamic learning rate scheduling to accelerate convergence.
- **Temperature Scaling**: Post-training confidence calibration.

#### ✅ Best Model Results (Saved at Epoch 47)

- **Validation Accuracy**: 64.53%
- **Weighted F1 Score**: 0.6415
- **Loss**: 0.6118

---
## Running via Streamlit
```bash
OPENCV_AVFOUNDATION_SKIP_AUTH=1 streamlit run streamlit/app.py
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