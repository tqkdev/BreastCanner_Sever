from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
from keras.models import load_model
import torch
from torchvision import transforms
import base64
from io import BytesIO


app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Chỉ định trang web gốc của bạn
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả các tiêu đề HTTP
)
# Tải mô hình phân đoạn ung thư vú
segmentor_model = load_model('./BreastCancerSegmentor.h5')

# Tải mô hình ResNet đã được fine-tuning
model_path = "./Resnet_fineTuning.pth"
model = torch.load(model_path)

# Định nghĩa các phép biến đổi
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def resize_image(contents, target_size):
    img = Image.open(io.BytesIO(contents))
    img = img.resize(target_size)
    return img


def image_to_base64(image):
    # Mã hóa dạng bytes thành base64
    base64_encoded = base64.b64encode(image).decode('utf-8')

    return base64_encoded


@app.get("/")
async def home():
    return 'sever is running'


@app.post("/predict")
async def predict(file: UploadFile):
    # Đọc hình ảnh đã tải lên
    contents = await file.read()
    # Thay đổi kích thước hình ảnh thành 256x256
    target_size = (256, 256)
    img = resize_image(contents, target_size)

    # Phân đoạn ung thư vú
    # img_array = img_to_array.img_to_array(img)
    img_array = np.array(img)
    img_proc = img_array / 255.0
    predictions = segmentor_model.predict(np.array([img_proc]))
    predicted_mask = Image.fromarray((predictions[0][:, :, 0] * 255).astype('uint8'))

    # Đảm bảo cả hai hình ảnh có chế độ giống nhau
    if img.mode != predicted_mask.mode:
        predicted_mask = predicted_mask.convert(img.mode)

    # Thay đổi kích thước hình ảnh nếu chúng không khớp
    if img.size != predicted_mask.size:
        img = img.resize(predicted_mask.size)

    # Kết hợp hai hình ảnh
    overlay = Image.blend(img, predicted_mask, alpha=0.8)

    # Áp dụng các phép biến đổi
    transformed_image = data_transforms(overlay)
    transformed_image = transformed_image.unsqueeze(0)

    # Tiến hành dự đoán
    model.eval()
    with torch.no_grad():
        output = model(transformed_image)

    # Lấy ra lớp dự đoán
    _, predicted = torch.max(output, 1)
    predicted_class = predicted.item()

    # Ánh xạ chỉ số lớp thành tên lớp
    class_names = ['benign', 'malignant', 'normal']
    predicted_class_name = class_names[predicted_class]

    # Chuyển hình overlay thành base64
    overlay_byte_array = BytesIO()
    overlay.save(overlay_byte_array, format="PNG")
    overlay_base64 = base64.b64encode(overlay_byte_array.getvalue()).decode()

    # Chuyển hình gốc thành base64
    img_byte_array = BytesIO()
    img.save(img_byte_array, format="PNG")
    img_base64 = base64.b64encode(img_byte_array.getvalue()).decode()

    return {"name_predict": predicted_class_name, "overlay_base64": overlay_base64, "img_base64": img_base64}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
