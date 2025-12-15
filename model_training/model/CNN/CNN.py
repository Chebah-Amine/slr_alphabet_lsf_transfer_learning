from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_cnn_model(input_shape):
    model = Sequential([
        # Couches Convolutionnelles avec régularisation L2
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Flatten(),

        # Couches Fully Connected avec régularisation L2
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),

        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),

        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),

        Dense(21, activation='softmax', kernel_regularizer=l2(0.01))
    ])

    # Compilation du modèle
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
