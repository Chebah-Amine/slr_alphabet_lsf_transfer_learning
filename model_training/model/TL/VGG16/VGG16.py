from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from TL.utils import freeze_layers

def pre_train_model_VGG16(input_shape=(64, 64, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # Geler les couches du modèle de base
    freeze_layers(base_model)
    return base_model

def create_model_VGG16(base_model):
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(21, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model