import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import gdown

st.set_page_config(page_title="App Nhận Diện Khuôn Mặt", layout="wide")

@st.cache_resource
def download_model():
    model_path = "face_model.h5"
    if not os.path.exists(model_path):
        file_id = '1s0JVa1Xa5KkMkDirqTobbFf9ZW5SOquJ'
        try:
            gdown.download(id=file_id, output=model_path, quiet=False)
        except Exception as e:
            st.error(f"Lỗi: {e}")
    return model_path

model_file_path = download_model()

@st.cache_resource
def load_trained_model(model_path=model_file_path):
    if os.path.exists(model_path):
        return load_model(model_path)
    return None
# ---------------------------------------------------------
# THANH CÔNG CỤ (SIDEBAR)
# ---------------------------------------------------------
st.sidebar.title("Thanh Công Cụ")
app_mode = st.sidebar.selectbox("Chọn chức năng", 
                                ["Dự đoán qua Webcam/Ảnh", "Huấn luyện mô hình (Training)"])

# ---------------------------------------------------------
# PHẦN 1: ỨNG DỤNG DỰ ĐOÁN (WEBCAM & UPLOAD)
# ---------------------------------------------------------
if app_mode == "Dự đoán qua Webcam/Ảnh":
    st.title("📷 Nhận diện Khuôn mặt (CNN)")
    
    # Load mô hình
    model = load_trained_model()
    
    if model is None:
        st.warning("⚠️ Chưa tìm thấy file mô hình 'face_model.h5'. Vui lòng qua mục 'Huấn luyện' để tạo mô hình trước, hoặc tải file model có sẵn lên thư mục GitHub!")
    else:
        st.success("✅ Đã tải mô hình thành công!")
        
        # Tạo 2 cột để bố trí UI đẹp hơn
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 1. Sử dụng Camera")
            camera_image = st.camera_input("Chụp ảnh từ Webcam")
            
        with col2:
            st.markdown("### 2. Tải ảnh từ máy")
            uploaded_file = st.file_uploader("Chọn file ảnh (jpg, png)", type=["jpg", "jpeg", "png"])
            
        # Xác định ảnh đầu vào
        img_to_predict = camera_image if camera_image else uploaded_file
        
        if img_to_predict is not None:
            # Hiển thị ảnh
            image = Image.open(img_to_predict).convert('RGB')
            st.image(image, caption="Ảnh đầu vào", use_column_width=True)
            
            # Tiền xử lý ảnh giống code gốc của bạn
            img_width, img_height = 200, 200
            img_resized = image.resize((img_width, img_height))
            img_array = np.array(img_resized)
            
            # Normalize pixel values to 0-1 và thêm batch dimension
            img_for_prediction = img_array.astype('float32') / 255.0
            img_for_prediction = np.expand_dims(img_for_prediction, axis=0)
            
            # Dự đoán
            if st.button("🔍 Tiến hành Dự đoán", type="primary"):
                with st.spinner("Đang phân tích..."):
                    preds = model.predict(img_for_prediction)
                    digit = np.argmax(preds)
                    confidence = np.max(preds) * 100
                    
                    st.success(f"**Kết quả dự đoán: Class {digit}**")
                    st.info(f"Độ tin cậy (Confidence): {confidence:.2f}%")

# ---------------------------------------------------------
# PHẦN 2: HUẤN LUYỆN MÔ HÌNH (GỮ NGUYÊN Ý CHÍNH CODE CŨ)
# ---------------------------------------------------------
elif app_mode == "Huấn luyện mô hình (Training)":
    st.title("⚙️ Huấn luyện Mô hình CNN")
    
    st.info("Nhập đường dẫn chứa dữ liệu huấn luyện. Lưu ý: Khi deploy lên GitHub, hãy dùng đường dẫn tương đối (ví dụ: `dataset/train`) thay vì ổ đĩa cục bộ hoặc Google Drive.")
    
    train_dir = st.text_input("Đường dẫn thư mục dữ liệu:", "data_khuon_mat/")
    epochs = st.number_input("Số lần học (Epochs):", min_value=1, max_value=100, value=10)
    batch_size = st.number_input("Batch Size:", min_value=8, max_value=128, value=32)
    
    if st.button("🚀 Bắt đầu Huấn luyện"):
        if not os.path.exists(train_dir):
            st.error(f"Không tìm thấy thư mục: {train_dir}. Vui lòng kiểm tra lại đường dẫn!")
        else:
            with st.spinner("Đang tiến hành huấn luyện... quá trình này có thể mất thời gian..."):
                img_width, img_height = 200, 200
                
                # Sửa lỗi: Thêm validation_split để vẽ được biểu đồ val_accuracy
                train_datagen = ImageDataGenerator(
                    rescale=1.0/255, 
                    rotation_range=30,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode="nearest",
                    validation_split=0.2 # Trích 20% data cho validation
                )                                                                                                                                                                       
                
                train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    class_mode="categorical",
                    subset='training'
                )
                
                val_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    class_mode="categorical",
                    subset='validation'
                )

                # XÂY DỰNG MÔ HÌNH CNN (Giữ nguyên cấu trúc của bạn)
                model = Sequential([
                    Conv2D(32, (3,3), activation="relu", input_shape=(img_width, img_height, 3)), 
                    MaxPooling2D(2,2),
                    Conv2D(64, (3,3), activation="relu"),
                    MaxPooling2D(2,2),
                    Conv2D(128, (3,3), activation="relu"),
                    MaxPooling2D(2,2),
                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(0.5),
                    Dense(22, activation="softmax") # Lưu ý: 22 là số class của bạn
                ])                                                                                                                                                                      
                
                model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                
                # In cấu trúc ra UI
                st.text("Cấu trúc mô hình:")
                model.summary(print_fn=lambda x: st.text(x))

                # HUẤN LUYỆN
                history = model.fit(
                    train_generator, 
                    validation_data=val_generator, # Sửa lỗi code cũ thiếu validation data
                    epochs=epochs
                )
                
                # Lưu mô hình
                model.save("face_model.h5")
                st.success("✅ Huấn luyện hoàn tất! Mô hình đã được lưu tại 'face_model.h5'. Bạn có thể chuyển sang tab Dự đoán để test.")

                # ĐÁNH GIÁ KẾT QUẢ MÔ HÌNH
                fig, ax = plt.subplots()
                ax.plot(history.history['accuracy'], label="Kết quả huấn luyện")
                ax.plot(history.history['val_accuracy'], label="Độ chính xác xác thực")
                ax.set_xlabel("Số lần học")
                ax.set_ylabel("Độ chính xác")
                ax.legend()
                st.pyplot(fig)
