from generate_data import generate_data_image, train_generator, validation_generator, test_generator
from evaluate_model import evaluate_model, plot_history
from utils import save_model, get_dataset_224x224
from TL.MobileNetV2.MobileNetV2 import pre_train_model_MobileNetV2, create_model_MobileNetV2
from train_model import train_model

train_dir, validation_dir, test_dir = get_dataset_224x224()
train_datagen, validation_datagen, test_datagen = generate_data_image()
train_generator = train_generator(train_datagen, train_dir, (224,224))
validation_generator = validation_generator(validation_datagen, validation_dir, (224,224))
test_generator = test_generator(test_datagen, test_dir, (224,224))
class_indices = train_generator.class_indices
classes = list(class_indices.keys())
print(classes)
model_base = pre_train_model_MobileNetV2(input_shape=(224, 224, 3))
model = create_model_MobileNetV2(model_base)
history = train_model(model, train_generator, validation_generator, 50)
evaluate_model(model, test_generator)
plot_history(history)
save_model(model, 'MobileNetV2_model_224x224.h5')