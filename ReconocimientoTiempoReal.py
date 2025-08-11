# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:32:38 2025

@author: AlbieriK
"""

import cv2 # Librería de python computer vision "CV"
import sys # Librería de python para interactuar con el sistema operativo.
           # en este caso para iniciar y cerrar la cámara :v

#url = 'http://10.11.3.166:8080/video' # IP y puerto que me dio la app webcam (app de teléfono)
#Biblioteca url = 'http://10.13.2.37:8080/video'
#UNI-IP  url = 'http://10.11.3.166:8080/video'
url = 'http://192.168.1.1:8080/video'
cap = cv2.VideoCapture(url) # función de CV para abrir una fuente de video (osea Cámara XD)

if not cap.isOpened():
    print("Inicia el servidor o checa tu IP y el puerto")
    sys.exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No detecte nada y me eh detenido ")
        break

    cv2.imshow('reconociendo...', frame)

#detener el programa
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27 or key == 32:
        # 'q' = 113, Esc = 27, Espacio = 32
        break

cap.release()
cv2.destroyAllWindows()
