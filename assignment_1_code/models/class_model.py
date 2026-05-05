import torch
import torch.nn as nn
from pathlib import Path


class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def save(self, save_dir: Path, suffix=None):
        """
        Saves the model, adds suffix to filename if given.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if suffix is None:
            filename = "model.pt"
        else:
            filename = f"model_{suffix}.pt"

        save_path = save_dir / filename

        torch.save(self.net.state_dict(), save_path)

        return save_path

    def load(self, path):
        """
        Loads model from path.
        Does not work with transfer model.
        """
        path = Path(path)

        if not path.is_file():
            raise ValueError(f"Model checkpoint does not exist: {path}")

        device = next(self.net.parameters()).device

        state_dict = torch.load(path, map_location=device)

        self.net.load_state_dict(state_dict)

        return self