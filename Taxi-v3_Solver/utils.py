import numpy as np

def save_model(model, path):
    """Save the Q-table model."""
    np.save(path, model)

def load_model(path):
    """Load a pre-trained Q-table model."""
    return np.load(path)
