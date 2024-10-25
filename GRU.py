import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(64)

# Load Tiny Shakespeare Data
# ------------------------------------------------------------------------------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
# -------------------------------------------------------------------------------------

# Model Variables
# ---------------------------------------------------------------------------------
vocab_size = len(itos)
block_size = 32
d_model = 24  # embedding dimension
n_hidden = 200
batch_size = 32


# ------------------------------------------------------------------------------------
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    if split == "train":
        data = train_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
    else:
        data = val_data
        ix = torch.arange(len(data) - block_size)
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
    return x, y


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = nn.Embedding(vocab_size, d_model)
        self.W = nn.Linear(d_model, 2 * n_hidden, bias=True)
        self.U = nn.Linear(n_hidden, 2 * n_hidden, bias=False)
        self.W_c = nn.Linear(d_model, n_hidden, bias=True)
        self.U_c = nn.Linear(n_hidden, n_hidden, bias=False)
        self.V = nn.Linear(n_hidden, vocab_size, bias=True)
        self.h_0 = nn.Parameter(0.1 * torch.ones(1, n_hidden, device=device))

    def forward(self, xb, targets):
        B, T = xb.shape
        emb = self.C(xb)  # [B, T, d_model]
        h_t = self.h_0.repeat(B, 1)  # [B, n_hidden]
        h_all = torch.zeros(B, T, n_hidden, device=device)
        for t in range(T):
            x_t = emb[:, t, :]  # [B, d_model]
            gates = torch.sigmoid(self.W(x_t) + self.U(h_t))  # [B, 2*n_hidden]
            z_t, r_t = gates.chunk(2, dim=1)  # both [B, n_hidden]
            c_t = torch.tanh(
                self.W_c(x_t) + self.U_c(r_t * h_t)
            )  # [B, n_hidden], c_t is h_tilda_t, the candidate content to be added
            h_t = (1 - z_t) * h_t + z_t * c_t
            h_all[:, t, :] = h_t
        logits = self.V(h_all)  # [B, T, vocab_size]
        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.shape[-1])  # [B*T, vocab_size]
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"Using device: {device}")

model = GRU().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

import math

max_lr = 1e-3  # Maximum learning rate
min_lr = max_lr * 0.1  # Minimum learning rate (10% of max_lr)
warmup_steps = 1000  # Number of warmup steps
max_steps = 10000  # Total number of training steps

def get_lr(it):
    if it < warmup_steps:
        # Linear warmup
        return max_lr * (it + 1) / warmup_steps
    elif it > max_steps:
        # After max_steps, keep learning rate constant at min_lr
        return min_lr
    else:
        # Cosine decay
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

optimizer = torch.optim.AdamW(model.parameters())
for step in range(max_steps):
    optimizer.zero_grad()
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if step % 1000 == 0:
        print(f"Step {step}: {loss.item()}")

# Inference
def evaluate(model):
    model.eval()
    total_loss = 0.0
    num_batches = 1000  # GPU memory -- average loss over 1000 batches from val data

    with torch.no_grad():  # Disable gradient computation
        for _ in range(num_batches):
            xb, yb = get_batch("val")
            logits, loss = model(xb, yb)
            total_loss += loss.item()

    average_loss = total_loss / num_batches
    print(f"Validation Loss: {average_loss}")
    model.train()

evaluate(model)

"""
python GRU.py
Using device: cuda
Total number of parameters: 149825
Step 0: 4.1863813400268555
Step 1000: 1.9502270221710205
Step 2000: 1.7134312391281128
Step 3000: 1.5254305601119995
Step 4000: 1.4830595254898071
Step 5000: 1.3990468978881836
Step 6000: 1.435714840888977
Step 7000: 1.4063057899475098
Step 8000: 1.4526458978652954
Step 9000: 1.4904130697250366
Validation Loss: 1.6415502626895904c
"""
