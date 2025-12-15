from TL.utils import summary_model
from TL.select_freezed_layers import select_policy_freezed_layers
from tensorflow.keras.applications import VGG16
from TL.VGG16.VGG16 import create_model_VGG16
from train_model import train_model
from utils import get_dataset_64x64, save_model
from generate_data import generate_data_image, train_generator, validation_generator, test_generator

def freeze_layers(model, layer_indices):
    for i, layer in enumerate(model.layers):
        layer.trainable = i in layer_indices

acc_strat = []
loss_strat = []

train_dir, validation_dir, test_dir = get_dataset_64x64()
train_datagen, validation_datagen, test_datagen = generate_data_image()

for freeze_strategy in ['first', 'middle', 'last']:
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3))
    layer_indices = select_policy_freezed_layers(vgg16, freeze_strategy)
    freeze_layers(vgg16, layer_indices)

    train_gene = train_generator(train_datagen, train_dir, (64, 64))
    validation_gen = validation_generator(validation_datagen, validation_dir, (64, 64))
    test_gen = test_generator(test_datagen, test_dir, (64, 64))

    model = create_model_VGG16(vgg16)
    history = train_model(model, train_generator, validation_generator, 30)
    test_loss, test_accuracy = model.evaluate(test_generator)
    acc_strat.append(test_accuracy)
    loss_strat.append(test_loss)
    save_model(model, f'/static/vgg16_model_64x64_{freeze_strategy}.h5')

    print(f'acc for strat {freeze_strategy} : {test_accuracy}')
    print(f'loss for strat {freeze_strategy} : {test_loss}')

print(f'all accuracies for first, middle and last strategies : {acc_strat}')
print(f'all losses for first, middle and last strategies : {loss_strat}')

# summary_model(vgg16)
