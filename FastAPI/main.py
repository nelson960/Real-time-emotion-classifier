from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import timm
from PIL import Image
import io 
from torchvision import transforms

app = FastAPI(title="Facial Emotion Recognition API")
emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'confused', 'calm']

model = timm.create_model("mobilevitv2_100", pretrained=False, num_classes=9)
model.load_state_dict(torch.load("/Users/nelson/py/paper_impl/emotion_classifier/models/best_model_ema_47.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
	transforms.Resize((256,256)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
	image = Image.open(io.BytesIO(await file.read())).convert("RGB")
	input_tensor = transform(image).unsqueeze(0)
	with torch.no_grad():
		output = model(input_tensor)
		probs = torch.nn.functional.softmax(output[0], dim=0)
	pred = torch.argmax(probs).item()
	return JSONResponse({
		"prediction": emotion_labels[pred],
		"probabilites": {emotion_labels[i]: float(p) for i, p in enumerate(probs)}
	})
