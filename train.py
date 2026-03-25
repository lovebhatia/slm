import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config   import cfg, device
from src.data     import BPETokenizer, DatasetBuilder, TextDataset
from src.model    import SLM
from src.utils    import CheckpointManager, TrainingLogger

def prepare_data(tokenizer_path: str = "models/tokenizer.json") -> TextDataset:
    """
    Loads or builds the training dataset.
    Returns a TextDataset ready for DataLoader.
    """
    # ── Step 1: Load or train the tokenizer ──────────────────
    tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)

    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print(f"Tokenizer loaded — vocab size: {tokenizer.vocab_size}")
    else:
        print("No tokenizer found — building from data/train.txt ...")
        if not os.path.exists("data/train.txt"):
            raise FileNotFoundError(
                "data/train.txt not found.\n"
                "Run data_collector.py first to generate training data."
            )
        with open("data/train.txt", encoding="utf-8") as f:
            raw_text = f.read()
        tokenizer.train(raw_text)
        tokenizer.save(tokenizer_path)

    # ── Step 2: Load training text and encode it ─────────────
    with open("data/train.txt", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Encoding {len(raw_text):,} characters ...")
    token_ids = tokenizer.encode(raw_text)
    print(f"Total tokens: {len(token_ids):,}")

    # ── Step 3: Create PyTorch Dataset ───────────────────────
    dataset = TextDataset(token_ids, cfg.context_len)
    print(f"Dataset samples: {len(dataset):,}")
    return dataset, tokenizer

@torch.no_grad()
def evaluate(model: SLM, dataloader: DataLoader, max_batches: int = 20) -> float:
    """
    Computes average loss over a subset of batches.
    @torch.no_grad() — no gradient tracking needed, saves memory.
    max_batches — we don't evaluate the whole dataset every time (too slow).
    Returns average loss as a float.
    """
    model.eval()   # disables dropout — deterministic output
    losses = []

    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())

    model.train()  # re-enable dropout for training
    avg_loss = sum(losses) / len(losses)
    return avg_loss
    
def train():
    """
    Full training loop — the heart of Phase 4.
    Runs for cfg.max_iters steps.
    """
    print("=" * 60)
    print(f"Device       : {device}")
    print(f"Batch size   : {cfg.batch_size}")
    print(f"Context len  : {cfg.context_len}")
    print(f"Max iters    : {cfg.max_iters}")
    print(f"Learning rate: {cfg.learning_rate}")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────
    dataset, tokenizer = prepare_data()

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,          # randomise order every epoch
        num_workers=0,         # 0 = main process (safe for MPS)
        pin_memory=False,      # MPS doesn't use pinned memory
        drop_last=True,        # drop incomplete final batch
    )

    # ── 2. Model ─────────────────────────────────────────────
    model = SLM().to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. Optimizer ─────────────────────────────────────────
    # Separate weight decay: apply to weight matrices, not biases/norms
    decay_params     = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() < 2]

    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8)

    # ── 4. Learning rate scheduler ───────────────────────────
    # Cosine annealing — starts at lr, decays to lr/10 smoothly
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_iters,
        eta_min=cfg.learning_rate / 10,
    )

    # ── 5. Checkpoint + Logger ───────────────────────────────
    ckpt   = CheckpointManager(cfg.checkpoint_dir)
    logger = TrainingLogger(cfg.log_dir)

    # Resume from checkpoint if one exists
    start_step, best_loss = ckpt.load(model, optimizer)

    # ── 6. Training loop ─────────────────────────────────────
    model.train()
    data_iter   = iter(dataloader)
    total_start = time.time()

    for step in range(start_step, cfg.max_iters):

        # Get next batch — cycle through dataset
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # ── Forward pass ──────────────────────────────────────
        logits, loss = model(x, targets=y)

        # ── Backward pass ─────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)  # more efficient than zero_grad()
        loss.backward()

        # ── Gradient clipping ─────────────────────────────────
        # Prevents gradient explosion — clips all gradients to max norm 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # ── Weight update ─────────────────────────────────────
        optimizer.step()
        scheduler.step()

        # ── Logging ───────────────────────────────────────────
        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            val_loss   = evaluate(model, dataloader)
            perplexity = math.exp(val_loss)
            current_lr = scheduler.get_last_lr()[0]

            logger.log(step, loss.item(), val_loss)

            elapsed = time.time() - total_start
            print(f"\nstep {step:5d} | "
                  f"train loss: {loss.item():.4f} | "
                  f"val loss: {val_loss:.4f} | "
                  f"perplexity: {perplexity:.1f} | "
                  f"lr: {current_lr:.6f} | "
                  f"time: {elapsed:.0f}s")

            # Save best checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                ckpt.save(model, optimizer, step, val_loss, tag="best")

        # Save latest checkpoint every 500 steps
        if step % 500 == 0 and step > 0:
            ckpt.save(model, optimizer, step, loss.item(), tag="latest")

    # ── End of training ───────────────────────────────────────
    logger.save()
    ckpt.save(model, optimizer, cfg.max_iters, best_loss, tag="final")

    total_time = time.time() - total_start
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Best perplexity: {math.exp(best_loss):.1f}")


if __name__ == "__main__":
    train()   
        
