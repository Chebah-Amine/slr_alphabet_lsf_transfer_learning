
from generate_data import generate_data_image, train_generator, validation_generator, test_generator
from evaluate_model import evaluate_model, plot_history
from utils import save_model, get_dataset_64x64
from TL.VGG16.VGG16 import create_model_VGG16, pre_train_model_VGG16
from train_model import train_model

train_dir, validation_dir, test_dir = get_dataset_64x64()
train_datagen, validation_datagen, test_datagen = generate_data_image()
train_generator = train_generator(train_datagen, train_dir, (64, 64))
validation_generator = validation_generator(validation_datagen, validation_dir, (64, 64))
test_generator = test_generator(test_datagen, test_dir, (64, 64))
model_base = pre_train_model_VGG16()
model = create_model_VGG16(model_base)
history = train_model(model, train_generator, validation_generator, 50)
evaluate_model(model, test_generator)
plot_history(history)
save_model(model, 'VGG16_2_model_64x64.h5')