# src/utils/checkpoint.py
import os
import torch


class CheckpointManager:
    """
    Saves and loads model checkpoints during training.
    Keeps the best checkpoint and the latest checkpoint separately.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints/"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, step: int, loss: float, tag: str = "latest"):
        path = os.path.join(self.checkpoint_dir, f"{tag}.pt")
        torch.save({
            "step"      : step,
            "loss"      : loss,
            "model"     : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved → {path}  (step {step}, loss {loss:.4f})")

    def load(self, model, optimizer, tag: str = "latest"):
        path = os.path.join(self.checkpoint_dir, f"{tag}.pt")
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return 0, float('inf')
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Checkpoint loaded ← {path}  (step {ckpt['step']}, loss {ckpt['loss']:.4f})")
        return ckpt["step"], ckpt["loss"]