def select_policy_freezed_layers(base_model, freeze_strategy='all'):
    if freeze_strategy == 'first':
        layer_indices = list(range(len(base_model.layers) // 3))
    elif freeze_strategy == 'middle':
        layer_indices = list(range(len(base_model.layers) // 3, 2 * len(base_model.layers) // 3))
    elif freeze_strategy == 'last':
        layer_indices = list(range(2 * len(base_model.layers) // 3, len(base_model.layers)))
    else:
        raise ValueError(f"Unknown freeze strategy: {freeze_strategy}")
    return layer_indices