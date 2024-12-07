import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import jax
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Sử dụng từ TensorFlow/Keras
import matplotlib.pyplot as plt

# Đường dẫn dữ liệu
path = os.getcwd()
raw_folder = os.path.join(path, "data")

# ---------------Start định dạng lại file và load ảnh-------------------
def save_data(raw_folder=raw_folder):
    dest_size = (128, 128)
    print("Bắt đầu xử lý ảnh...")

    pixels = []
    labels = []

    if not os.path.exists(raw_folder):
        print(f"Thư mục {raw_folder} không tồn tại!")
        return

    for folder in os.listdir(raw_folder):
        folder_path = os.path.join(raw_folder, folder)
        if os.path.isdir(folder_path):
            print(f"Folder: {folder}")
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    img = cv2.imread(file_path)
                    if img is not None:
                        img_resized = cv2.resize(img, dsize=dest_size)
                        pixels.append(img_resized)
                        labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels)

    # One-hot encoding
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)

    # Lưu dữ liệu
    with open("pix.data", "wb") as file:
        pickle.dump((pixels, labels), file)

    print(f"Dữ liệu đã được lưu! {len(pixels)} ảnh.")
    return

def load_data():
    if not os.path.exists("pix.data"):
        print("Không tìm thấy file pix.data!")
        return None, None

    with open("pix.data", "rb") as file:
        pixels, labels = pickle.load(file)

    print(f"Dữ liệu đã load: {pixels.shape}, {labels.shape}")
    return pixels, labels

save_data()
X, Y = load_data()
if X is None or Y is None:
    exit()

# ----------Chia thành 2 biến data train và test------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
print(X_train.shape, Y_train.shape)

# -------------Cấu hình mô hình CNN theo cấu trúc VGG16 -------------
def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    input_layer = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input_layer)

    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)

    my_model = Model(inputs=input_layer, outputs=x)
    my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

vggmodel = get_model()

# -------------Cấu hình Tiền xử lý dữ liệu -------------
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.1,
    rescale=1.0 / 255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.2, 1.5],
    fill_mode="nearest",
)

# -------------Huấn luyện mô hình CNN-------------
filepath = os.path.join(path, "weights-{epoch:02d}-{val_accuracy:.2f}.weights.h5")
checkpoint = ModelCheckpoint(filepath,save_weights_only=True, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

vgghist = vggmodel.fit(
    aug.flow(X_train, Y_train, batch_size=64),
    epochs=50,
    validation_data=aug.flow(X_test, Y_test, batch_size=len(X_test)),
    callbacks=callbacks_list,
)

vggmodel.save(os.path.join(path, "vggmodel.keras"))
