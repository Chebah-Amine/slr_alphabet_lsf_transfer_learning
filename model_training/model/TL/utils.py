def freeze_layers(base_model):
    for layer in base_model.layers:
        layer.trainable = False

def summary_model(base_model):
    base_model.summary()