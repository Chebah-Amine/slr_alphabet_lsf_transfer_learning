from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, train_generator, validation_generator, train_epochs, callback=True):
    # Callback pour Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    if callback:
        history = model.fit(
            train_generator,
            epochs=train_epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping]
        )
    else : 
        history = model.fit(
            train_generator,
            epochs=train_epochs,
            validation_data=validation_generator,
        )
    return history