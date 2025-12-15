from validate_model import cross_validate_model
from utils import get_dataset_cross_validation_224x224
from generate_data import generate_data_image, train_generator
from TL.MobileNetV2.MobileNetV2 import pre_train_model_MobileNetV2, create_model_MobileNetV2

def create_model_func():
    model_base = pre_train_model_MobileNetV2(input_shape=(224, 224, 3))
    model = create_model_MobileNetV2(model_base)
    return model

dimension = (224, 224)

dataset_dir = get_dataset_cross_validation_224x224()
train_datagen, validation_datagen, test_datagen = generate_data_image()
generator = train_generator(train_datagen, dataset_dir, dimension)
cross_validate_model(create_model_func, train_datagen, validation_datagen, generator, dimension)