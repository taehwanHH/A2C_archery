from datetime import datetime

Hyper_Param = {
    'today': datetime.now().strftime('%Y-%m-%d'),
    'discount_factor': 0.95,
    'beta': 1,
    'beta_min': 0.0001,
    'beta_decay_rate': 0.9998,
    'learning_rate': 0.0005,
    'batch_size': 300,
    'num_episode': 200000,
    'print_every': 1000,
    'num_neurons': [32,64,64,64,32],
    'step_max': 300,
    'vw_max': 5,
    'window_size': 1000,
    'Saved_using': False,
    'MODEL_PATH': "saved_model",
    'MODEL_NAME': "model_(227, 1001.0).h5"
}

