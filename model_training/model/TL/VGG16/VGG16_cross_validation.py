from validate_model import cross_validate_model
from utils import get_dataset_cross_validation_64x64
from generate_data import generate_data_image, train_generator
from TL.VGG16.VGG16 import create_model_VGG16, pre_train_model_VGG16

def create_model_func():
    model_base = pre_train_model_VGG16()
    model = create_model_VGG16(model_base)
    return model

dimension = (64, 64)

dataset_dir = get_dataset_cross_validation_64x64()
train_datagen, validation_datagen, test_datagen = generate_data_image()
generator = train_generator(train_datagen, dataset_dir, dimension)
cross_validate_model(create_model_func, train_datagen, validation_datagen, generator, dimension)