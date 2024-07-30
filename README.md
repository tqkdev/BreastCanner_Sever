#  FastAPI 

## Giới Thiệu

Đây là backend của ứng dụng học máy dự đoán bệnh được xây dựng bằng FastAPI

## Mô tả
- Sử dụng mạng nơron CNN để train một mô hình học máy dự đoán bệnh => lưu vào file Resnet_fineTuning.pth và BreastCancerSegmentor.h5
- Backend nhận ảnh từ Frontend và tiến hành dự đoán bằng học máy đã train ở trên
- Trả kết quả về cho Frontend

## Công Nghệ
- **Backend**: FastAPI , Python.
