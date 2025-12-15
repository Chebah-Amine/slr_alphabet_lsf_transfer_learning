from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_data_image(): 
    # Générateurs d'images avec augmentation de données pour l'entraînement
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    return train_datagen, validation_datagen, test_datagen

def train_generator(train_datagen, train_dir, target_size, batch_size=64, grayscale=False):

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    if grayscale:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale'
        )
    return train_generator

def validation_generator(validation_datagen, validation_dir, target_size, batch_size=64, grayscale=False):
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    if grayscale:
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale'
        )
    return validation_generator

def test_generator(test_datagen, test_dir, target_size, batch_size=64, grayscale=False):
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    if grayscale:
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=False
        )

    return test_generator