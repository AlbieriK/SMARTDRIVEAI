# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 22:44:02 2025

@author: dell

"""

                               #Librerias
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

model = load_model('car_classifier_audi_toyota.h5')

                        
                                #RUTAS
#--------------------------------------------------------------------------------- 
                                                                               # |Ruta principal
base_dir = r'C:\Users\dell\Desktop\Datasetcars\archive\Cars Dataset'           # |
                                                                               # |
                                                                               # |
train_dir = os.path.join(base_dir, 'train')                                    # |Ruta entrenamiento 80%
                                                                               # |
                                                                               # |
                                                                               # |
test_dir = os.path.join(base_dir, 'test')                                      # |Ruta Prueba 20%
#---------------------------------------------------------------------------------

# Paso 3: Preparar generadores de datos
train_datagen = ImageDataGenerator(rescale=1./255)  # normaliza entre 0 y 1
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # tamaño de las imágenes
    batch_size=16,
    class_mode='binary'  # dos clases: Audi y Toyota Innova
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,  # cambio aquí para que lea la carpeta 'test'
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

# Paso 4: Crear el modelo CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binaria
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Paso 5: Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=10,  # prueba con 10 epochs, luego ajustas
    validation_data=validation_generator
)

# Paso 6: Guardar el modelo
model.save(r'C:\Users\dell\SmartDriveAI Proyecto\Backend\car_classifier_audi_toyota.h5')


print("Me eh entrenado correctamente y me guarde como 'car_classifier_audi_toyota.h5'")

# Paso 7: Graficar accuracy y loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# Gráfica de precisión
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Entrenamiento')
plt.plot(epochs_range, val_acc, label='Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Gráfica de pérdida
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Entrenamiento')
plt.plot(epochs_range, val_loss, label='Validación')
plt.title('Perdida durante el entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Perdida')
plt.legend()

plt.tight_layout()
plt.show()
