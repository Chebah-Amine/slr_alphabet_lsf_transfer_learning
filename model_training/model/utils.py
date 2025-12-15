def save_model(model, path):
    # Sauvegarder le modèle
    model.save(path)

def get_dataset_64x64():
    # Chemins vers les datasets
    train_dir = 'C:/Users/amine/COURS/NANTERRE/M2/MEMOIRE/PROJET/MEMOIRE M2/DATASET_REDIM_64x64/train/'
    validation_dir = 'C:/Users/amine/COURS/NANTERRE/M2/MEMOIRE/PROJET/MEMOIRE M2/DATASET_REDIM_64x64/validation/'
    test_dir = 'C:/Users/amine/COURS/NANTERRE/M2/MEMOIRE/PROJET/MEMOIRE M2/DATASET_REDIM_64x64/test/'

    return train_dir, validation_dir, test_dir

def get_gray_dataset_64x64():
    # Chemins vers les datasets
    train_dir = 'C:/Users/amine/COURS/NANTERRE/M2/MEMOIRE/PROJET/MEMOIRE M2/DATASET_REDIM_64x64_GRAY/train/'
    validation_dir = 'C:/Users/amine/COURS/NANTERRE/M2/MEMOIRE/PROJET/MEMOIRE M2/DATASET_REDIM_64x64_GRAY/validation/'
    test_dir = 'C:/Users/amine/COURS/NANTERRE/M2/MEMOIRE/PROJET/MEMOIRE M2/DATASET_REDIM_64x64_GRAY/test/'

    return train_dir, validation_dir, test_dir

def get_dataset_224x224():
    # Chemins vers les datasets
    train_dir = 'C:/Users/amine/COURS/NANTERRE/M2/MEMOIRE/PROJET/MEMOIRE M2/DATASET_REDIM_224x224/train/'
    validation_dir = 'C:/Users/amine/COURS/NANTERRE/M2/MEMOIRE/PROJET/MEMOIRE M2/DATASET_REDIM_224x224/validation/'
    test_dir = 'C:/Users/amine/COURS/NANTERRE/M2/MEMOIRE/PROJET/MEMOIRE M2/DATASET_REDIM_224x224/test/'

    return train_dir, validation_dir, test_dir

def get_dataset_cross_validation_64x64():
    return 'C:/Users/amine/COURS/NANTERRE/M2/MEMOIRE/PROJET/MEMOIRE M2/BACKUP/DATASET_REDIM_64x64'

def get_dataset_cross_validation_224x224():
    return 'C:/Users/amine/COURS/NANTERRE/M2/MEMOIRE/PROJET/MEMOIRE M2/BACKUP/DATASET_REDIM_224x224'