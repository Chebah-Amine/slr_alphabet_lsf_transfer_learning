import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def cross_validate_model(create_model_func, train_datagen, validation_datagen, generator, target_size, epochs=30):
    # Récupérer les fichiers et les étiquettes de toutes les images de formation
    filepaths = np.array(generator.filepaths)
    labels = np.array(generator.classes)

    # Placeholder pour les scores de chaque fold
    fold_val_accuracies = []
    fold_val_losses = []

    # Définir StratifiedKFold avec 5 folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    
    # Boucle sur chaque fold
    for train_index, val_index in skf.split(filepaths, labels):
        # Diviser les fichiers et les étiquettes en ensembles d'entraînement et de validation
        train_files, val_files = filepaths[train_index], filepaths[val_index]
        train_labels, val_labels = labels[train_index].astype(str), labels[val_index].astype(str)

        # Afficher la répartition des classes pour le fold courant
        print(f'Train fold class distribution: {np.bincount(train_labels.astype(int))}')
        print(f'Validation fold class distribution: {np.bincount(val_labels.astype(int))}')

        # Créer les générateurs pour le fold actuel
        train_fold_gen = train_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({'filename': train_files, 'class': train_labels}),
            directory=None,
            x_col='filename',
            y_col='class',
            target_size=target_size,
            batch_size=64,
            class_mode='categorical'
        )
        
        val_fold_gen = validation_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({'filename': val_files, 'class': val_labels}),
            directory=None,
            x_col='filename',
            y_col='class',
            target_size=target_size,
            batch_size=64,
            class_mode='categorical'
        )
        
        # Créer un nouveau modèle pour chaque fold
        model_cv = create_model_func()
        
        # Callback pour Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Entraîner le modèle
        history = model_cv.fit(
            train_fold_gen,
            epochs=epochs,
            validation_data=val_fold_gen,
            callbacks=[early_stopping]
        )
        
        # Enregistrer les scores de validation
        best_val_accuracy = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])
        fold_val_accuracies.append(best_val_accuracy)
        fold_val_losses.append(best_val_loss)
        
        print(f'Fold validation accuracy: {best_val_accuracy:.4f}, Fold validation loss: {best_val_loss:.4f}')
    
    # Afficher les résultats moyens de validation
    print(f'Mean validation accuracy: {np.mean(fold_val_accuracies):.4f}')
    print(f'Mean validation loss: {np.mean(fold_val_losses):.4f}')
    print(f'acc all folders {fold_val_accuracies}')
    print(f'loss all folders {fold_val_losses}')