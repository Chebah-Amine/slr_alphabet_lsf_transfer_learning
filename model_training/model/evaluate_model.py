import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

def evaluate_model(model, test_generator):
    # Évaluation du modèle
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test accuracy: {test_accuracy:.2f}')
    print(f'Test loss: {test_loss:.2f}')

    # Faire des prédictions sur les données de test
    test_generator.reset()
    predictions = model.predict(test_generator)

    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
    print_confusion_matrix(y_pred, y_true, test_generator)


# Tracer les courbes de perte et de précision
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Tracer la courbe de précision
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Tracer la courbe de perte
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def print_confusion_matrix(y_pred, y_true, test_generator):
    # Générer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)

    # Visualiser la matrice de confusion
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()