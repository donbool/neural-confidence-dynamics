# fixed_points.py

import torch
import numpy as np
from scipy.optimize import fsolve
from model import ContextRNN

def find_fixed_point(model, input_vec, h_init, tol=1e-6, max_iter=100):
    """
    Finds a fixed point h* such that h* = RNN(h*, x).
    """

    input_tensor = torch.tensor(input_vec, dtype=torch.float32).view(1, 1, -1)  # shape: (1, 1, input_dim)

    def fn(h_numpy):
        h_tensor = torch.tensor(h_numpy, dtype=torch.float32).view(1, 1, -1)  # shape: (1, 1, hidden_dim)
        _, h_next = model.rnn(input_tensor, h_tensor)
        return (h_next.view(-1).detach().numpy() - h_numpy)

    h_star = fsolve(fn, h_init, xtol=tol, maxfev=max_iter)
    return h_star

if __name__ == "__main__":
    # Load trained model
    model = ContextRNN(input_size=3, hidden_size=64, output_size=2)
    model.load_state_dict(torch.load("trained_model.pth", map_location="cpu"))
    model.eval()

    # Choose a fixed input context
    x = np.array([1.0, 0.0, 0.0])  # example input

    # Try finding a fixed point starting from random initial state
    h0 = np.random.randn(64)  # random init for hidden
    fp = find_fixed_point(model, x, h0)

    print("Fixed point found:", fp.round(4))
