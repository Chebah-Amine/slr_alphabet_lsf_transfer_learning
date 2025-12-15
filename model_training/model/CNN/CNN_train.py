from generate_data import generate_data_image, train_generator, validation_generator, test_generator
from evaluate_model import evaluate_model, plot_history
from utils import save_model, get_dataset_64x64
from CNN.CNN import create_cnn_model
from train_model import train_model

train_dir, validation_dir, test_dir = get_dataset_64x64()
train_datagen, validation_datagen, test_datagen = generate_data_image()
train_generator = train_generator(train_datagen, train_dir, (64, 64))
validation_generator = validation_generator(validation_datagen, validation_dir, (64, 64))
test_generator = test_generator(test_datagen, test_dir, (64, 64))
model = create_cnn_model((64, 64, 3))
history = train_model(model, train_generator, validation_generator, 50)
evaluate_model(model, test_generator)
plot_history(history)
save_model(model, 'cnn_model_64x64.h5')
