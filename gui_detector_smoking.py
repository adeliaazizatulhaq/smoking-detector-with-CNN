# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from imutils.video import VideoStream
from tkinter import Tk, Button, filedialog
from torchvision.models import resnet50  # Menggunakan ResNet50
import torch.nn as nn

# Argument parser untuk filter deteksi wajah
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--face_confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Memuat detektor wajah
print("[INFO] Loading face detector...")
protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Informasi awal
print("[INFO] Loading smoking detector...")

# Tentukan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Muat arsitektur model
model = resnet50(pretrained=False)
model.fc = nn.Linear(2048, 1)  # Output layer: 1 neuron untuk binary classification

# Muat state dict
state_dict = torch.load("detectorsmoking3.pth", map_location=device)
state_dict.pop('fc.weight', None)  # Hapus parameter fc jika ada konflik
state_dict.pop('fc.bias', None)

# Load state dict ke model
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

print("[INFO] Model berhasil dimuat dan siap untuk evaluasi.")

# Preprocessing pipeline untuk gambar
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Fungsi untuk memproses video (kamera atau file video)
def process_video(video_source):
    print("[INFO] Starting video stream...")
    if isinstance(video_source, str):  # File video
        vs = cv2.VideoCapture(video_source)
    else:  # Kamera
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

    while True:
        # Ambil frame dari video stream
        frame = vs.read() if isinstance(video_source, int) else vs.read()[1]
        if frame is None:
            break
        frame = imutils.resize(frame, width=600)

        # Ambil dimensi frame dan konversikan ke blob untuk deteksi wajah
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # Proses blob melalui network untuk mendapatkan deteksi wajah
        net.setInput(blob)
        detections = net.forward()

        # Loop melalui deteksi wajah
        for i in range(0, detections.shape[2]):
            # Ambil confidence dari prediksi
            confidence = detections[0, 0, i, 2]

            # Filter deteksi yang lemah
            if confidence > args["face_confidence"]:
                # Ambil koordinat (x, y) untuk bounding box wajah
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Pastikan bounding box tetap berada dalam dimensi frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # Ekstrak wajah dari ROI dan preprocessing
                face = frame[startY:endY, startX:endX]
                face = transform(face)
                face = face.unsqueeze(0)  # Menambahkan dimensi batch
                face = face.to(device)

                # Pass wajah melalui model deteksi merokok
                with torch.no_grad():
                    output = model(face)
                    predicted = torch.sigmoid(output).item()  # Konversi ke probabilitas
                    label = "Smoking" if predicted > 0.5 else "Not Smoking"

                # Gambar label dan bounding box pada frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)

        # Menampilkan frame secara realtime
        cv2.imshow("SMOKING_DETECTOR", frame)

        # Jika tombol 'q' ditekan, keluar dari loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Menutup semua window dan menghentikan video stream
    if isinstance(video_source, str):
        vs.release()
    else:
        vs.stop()
    cv2.destroyAllWindows()

# Fungsi untuk membuka kamera
def open_camera():
    process_video(0)

# Fungsi untuk memilih file video
def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        process_video(file_path)

# Fungsi untuk memilih file gambar
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    if file_path:
        # Membaca gambar yang diunggah
        image = cv2.imread(file_path)
        image = imutils.resize(image, width=600)

        # Ambil dimensi frame dan konversikan ke blob untuk deteksi wajah
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # Proses blob melalui network untuk mendapatkan deteksi wajah
        net.setInput(blob)
        detections = net.forward()

        # Loop melalui deteksi wajah
        for i in range(0, detections.shape[2]):
            # Ambil confidence dari prediksi
            confidence = detections[0, 0, i, 2]

            if confidence > args["face_confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # Ekstrak wajah dari ROI dan prediksi
                face = image[startY:endY, startX:endX]
                face = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(face)
                    predicted = torch.sigmoid(output).item()
                    label = "Smoking" if predicted < 0.51 else "Not Smoking"

                cv2.putText(image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

        cv2.imshow("Smoking Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Membuat antarmuka pengguna dengan tkinter
root = Tk()
root.title("Smoking Detector")
root.geometry("300x200")

btn_camera = Button(root, text="Open Camera", command=open_camera, width=20, height=2)
btn_camera.pack(pady=10)

btn_upload = Button(root, text="Upload Video", command=upload_video, width=20, height=2)
btn_upload.pack(pady=10)

btn_upload_image = Button(root, text="Upload Image", command=upload_image, width=20, height=2)
btn_upload_image.pack(pady=10)

root.mainloop()
