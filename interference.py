import os
import torch
import math
from src.config import cfg, device
from src.model  import SLM
from src.data   import BPETokenizer

# ── Helper: load model from checkpoint ──────────────────────

def load_model(checkpoint_path: str = "checkpoints/best.pt") -> SLM:
    """
    Loads a trained SLM from a checkpoint file.
    Returns the model in eval mode ready for inference.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}\n"
            f"Run train.py first to train the model."
        )
    
    model = SLM().to(device)
    
    # Load saved weights — map_location handles device differences
    # e.g. checkpoint saved on GPU, loading on CPU — no crash
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    
    model.eval()    # disable dropout — deterministic output
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Trained for   : {ckpt['step']} steps")
    print(f"  Best val loss : {ckpt['loss']:.4f}")
    print(f"  Parameters    : {sum(p.numel() for p in model.parameters()):,}")
    return model

# ── Helper: load tokenizer ───────────────────────────────────
def load_tokenizer(path: str = "models/tokenizer.json") -> BPETokenizer:
    
    """Loads the trained BPE tokenizer from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No tokenizer found at {path}\n"
            f"Run train.py first — it saves the tokenizer automatically."
        )
    tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)
    tokenizer.load(path)
    return tokenizer

# ── Core: generate text from a prompt ───────────────────────
def generate(
    model      : SLM,
    tokenizer  : BPETokenizer,
    prompt     : str,
    max_tokens : int   = 200,
    temperature: float = 0.8,
    top_k      : int   = 40,
) -> str:
    """
    Generates text continuation given a prompt string.

    Args:
        prompt      : the starting text e.g. "The capital of India is"
        max_tokens  : how many new tokens to generate
        temperature : 1.0 = normal, <1.0 = focused, >1.0 = creative
        top_k       : only sample from the top k most likely tokens
    """
    model.eval()

    # Encode prompt to token IDs
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(
        [input_ids], dtype=torch.long, device=device
    )   # shape: (1, seq_len)

    print(f"\nPrompt tokens : {len(input_ids)}")

    generated_ids = []

    for _ in range(max_tokens):

        # Crop to context window if needed
        context = input_tensor[:, -cfg.context_len:]

        # Forward pass — get logits for next token
        logits, _ = model(context)

        # Take logits at the LAST position — this is the next token prediction
        next_logits = logits[:, -1, :]   # shape: (1, vocab_size)

        # ── Temperature scaling ───────────────────────────────
        # Divide by temperature before softmax
        # Low temp (0.3) → sharper distribution → more predictable output
        # High temp (1.5) → flatter distribution → more random output
        next_logits = next_logits / temperature

        # ── Top-k filtering ───────────────────────────────────
        # Zero out all tokens except the top k most likely ones
        # This prevents the model from picking bizarre low-probability tokens
        if top_k > 0:
            # Get the k-th largest value as a threshold
            top_k_values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            threshold = top_k_values[:, -1].unsqueeze(-1)
            # Set everything below threshold to -infinity
            next_logits = next_logits.masked_fill(
                next_logits < threshold, float('-inf')
            )

        # ── Sample from distribution ──────────────────────────
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)   # shape: (1, 1)

        # Stop if model generates EOS token
        if next_token.item() == tokenizer.SPECIAL_TOKENS.get("<EOS>", 3):
            break

        generated_ids.append(next_token.item())

        # Append new token to the running context
        input_tensor = torch.cat([input_tensor, next_token], dim=1)

    # Decode generated tokens back to text
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


# ── Evaluation: test the model on exam questions ─────────────
def evaluate_on_exam_questions(model: SLM, tokenizer: BPETokenizer):
    """
    Runs the model on a set of exam-style prompts.
    Shows what the model has learned.
    """
    test_prompts = [
        # CAT questions
        "The capital of India is",
        "Quantitative aptitude problems include",
        "The Indian Parliament consists of",
        "Time speed and distance",
        "Simple interest is calculated as",

        # UPSC questions
        "The Indian Constitution was adopted on",
        "Fundamental Rights are guaranteed under",
        "The Non Cooperation Movement was launched by",
        "The Dandi March was undertaken by Gandhi",
        "Directive Principles of State Policy are",

        # SSC questions
        "The national animal of India is",
        "The Reserve Bank of India was established",
        "GDP stands for",
    ]

    print("\n" + "=" * 60)
    print("MODEL EVALUATION — EXAM QUESTION COMPLETIONS")
    print("=" * 60)

    for prompt in test_prompts:
        print(f"\nPrompt  : {prompt}")
        result = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=60,
            temperature=0.7,
            top_k=30,
        )
        print(f"Answer  : {result}")
        print("-" * 40)


# ── Interactive mode — chat with the model ───────────────────
def interactive_mode(model: SLM, tokenizer: BPETokenizer):
    """
    REPL loop — type prompts, get completions.
    Type 'quit' to exit, 'settings' to adjust parameters.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE — type a prompt, press Enter")
    print("Commands: 'quit' to exit | 'eval' to run exam test")
    print("=" * 60)

    # Default generation settings
    settings = {
        "max_tokens" : 150,
        "temperature": 0.8,
        "top_k"      : 40,
    }

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            print("Goodbye!")
            break
        if prompt.lower() == "eval":
            evaluate_on_exam_questions(model, tokenizer)
            continue

        # Generate and print
        print("Model: ", end="", flush=True)
        result = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
            top_k=settings["top_k"],
        )
        print(result)
        print(f"\n[tokens generated · temp={settings['temperature']} · top_k={settings['top_k']}]")


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model and tokenizer...")

    model     = load_model("checkpoints/best.pt")
    tokenizer = load_tokenizer("models/tokenizer.json")

    print(f"\nDevice: {device}")

    # First — run the automatic evaluation
    evaluate_on_exam_questions(model, tokenizer)

    # Then — drop into interactive mode
    interactive_mode(model, tokenizer)

## What every key piece does
"""
`map_location=device` — when you save a checkpoint on one device (GPU) and load on another (CPU), this handles the conversion automatically. Without it you get a crash.

`model.eval()` — turns off dropout. During training dropout randomly silences 10% of neurons to prevent memorisation. During inference you want deterministic, full-power output.

`@torch.no_grad()` — tells PyTorch not to build the computation graph during generation. Training needs the graph for backpropagation. Inference does not. Skipping it saves 30–40% memory and speeds up generation.

`temperature` — before softmax, divide all logits by temperature. Think of it like this:
```
temperature = 0.3  →  differences between logits get amplified
                       model picks the most likely token almost always
                       output: focused, repetitive, "safe"

temperature = 1.0  →  logits unchanged
                       output: balanced

temperature = 1.5  →  differences between logits get compressed
                       lower probability tokens get more chances
                       output: creative, varied, sometimes weird
"""
    