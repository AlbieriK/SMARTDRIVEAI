# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 19:44:02 2025

@author: AlbieriK
"""

import cv2
import numpy as np
from threading import Thread, Lock
from tensorflow.keras.models import load_model

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.lock = Lock()
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else None

    def stop(self):
        self.running = False
        self.cap.release()

def preprocess(frame):
    img = cv2.resize(frame, (150, 150))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main():
    # Carga del modelo
    model = load_model(r'C:\Users\dell\SmartDriveAI Proyecto\Backend\car_classifier_audi_toyota.h5')

    # Define la URL del stream de la cámara IP
    url = 'http://192.168.1.100:8080/video'  # Cambia esta IP por la tuya

    stream = VideoStream(url)  # Crea e inicia el video stream

    while True:
        ret, frame = stream.read()
        if not ret:
            print("No se recibió frame. Revisa la conexión a la cámara IP.")
            break

        img = preprocess(frame)
        pred = model.predict(img)[0][0]

        if pred < 0.4:
            label = "Audi"
        elif pred > 0.6:
            label = "Toyota Innova"
        else:
            label = "No te muevas"

        cv2.putText(frame, f"{label}: {pred:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('SmartAI', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
