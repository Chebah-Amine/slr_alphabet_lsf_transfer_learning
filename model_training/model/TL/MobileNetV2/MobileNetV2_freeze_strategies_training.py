from TL.utils import summary_model
from tensorflow.keras.applications import MobileNetV2
from generate_data import generate_data_image, train_generator, validation_generator, test_generator
from train_model import train_model
from utils import get_dataset_224x224, save_model
from TL.select_freezed_layers import select_policy_freezed_layers
from TL.MobileNetV2.MobileNetV2 import create_model_MobileNetV2


# summary_model(mobileNetV2)

def freeze_layers(model, layer_indices):
    for i, layer in enumerate(model.layers):
        layer.trainable = i in layer_indices

acc_strat = []
loss_strat = []

train_dir, validation_dir, test_dir = get_dataset_224x224()
train_datagen, validation_datagen, test_datagen = generate_data_image()

for freeze_strategy in ['first', 'middle', 'last']:
    mobileNetV2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    layer_indices = select_policy_freezed_layers(mobileNetV2, freeze_strategy)
    freeze_layers(mobileNetV2, layer_indices)

    train_gen = train_generator(train_datagen, train_dir, (224, 224))
    validation_gen = validation_generator(validation_datagen, validation_dir, (224, 224))
    test_gen = test_generator(test_datagen, test_dir, (224, 224))

    model = create_model_MobileNetV2(mobileNetV2)
    history = train_model(model, train_gen, validation_gen, 30)
    test_loss, test_accuracy = model.evaluate(test_gen)
    acc_strat.append(test_accuracy)
    loss_strat.append(test_loss)
    save_model(model, f'static/mobileNetV2_model_224x224_{freeze_strategy}.h5')

    print(f'acc for strat {freeze_strategy} : {test_accuracy}')
    print(f'loss for strat {freeze_strategy} : {test_loss}')

print(f'all accuracies for first, middle and last strategies : {acc_strat}')
print(f'all losses for first, middle and last strategies : {loss_strat}')