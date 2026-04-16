"""
Complete training pipeline for the MiniGPT model
Step-by-step guide to train with the new dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import json
from datetime import datetime

# ==================== STEP 1: TOKENIZER ====================
def train_tokenizer_on_new_data():
    """Train tokenizer on the combined dataset"""
    from tokenizers import ByteLevelBPETokenizer
    
    print("📝 Step 1: Training Tokenizer...")
    
    # Check which data files exist
    data_files = []
    if os.path.exists("combined_training_data.txt"):
        data_files.append("combined_training_data.txt")
        print("   Using: combined_training_data.txt (largest dataset)")
    elif os.path.exists("training_corpus.txt"):
        data_files.append("training_corpus.txt")
        print("   Using: training_corpus.txt")
    else:
        data_files.append("data.txt")
        print("   Using: data.txt (default)")
    
    os.makedirs("tokenizer", exist_ok=True)
    
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=data_files, vocab_size=30000)
    tokenizer.save_model("tokenizer")
    
    print(f"✅ Tokenizer trained! Vocab size: 30000")
    return tokenizer

# ==================== STEP 2: DATA LOADING ====================
class TextDataset(Dataset):
    """Accurate dataset for loading and tokenizing text"""
    
    def __init__(self, filename, tokenizer, max_length=512, chunk_size=1000000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"📂 Loading dataset: {filename}")
        
        file_size_mb = os.path.getsize(filename) / 1024 / 1024
        print(f"   File size: {file_size_mb:.1f} MB")
        
        import array
        self.tokens = array.array('i')
        print("   Tokenizing dataset in memory-efficient chunks...")
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                chunk = f.read(5 * 1024 * 1024)  # 5 MB chunks
                if not chunk:
                    break
                self.tokens.extend(tokenizer.encode(chunk).ids)
        
        self.num_sequences = max(0, (len(self.tokens) - 1) // self.max_length)
        print(f"   Total tokens: {len(self.tokens):,}")
        print(f"   Total sequences (training examples): {self.num_sequences:,}")
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.max_length
        chunk = list(self.tokens[start_idx : start_idx + self.max_length + 1])
        
        if len(chunk) < self.max_length + 1:
            chunk = chunk + [0] * (self.max_length + 1 - len(chunk))
            
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, target_ids

# ==================== STEP 3: MODEL ====================
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, heads, layers, max_seq_len=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.fc(x)
        return x

# ==================== STEP 4: TRAINING ====================
def train_model(
    model_name="minigpt_model",
    data_file="combined_training_data.txt",
    num_epochs=3,
    batch_size=4,  # Reduced default batch size
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_length=256,  # Reduced sequence length
    chunk_size=500000  # Load file in 500KB chunks
):
    """Full training pipeline with memory optimization"""
    
    print(f"\n{'='*60}")
    print("🚀 MINIGPT TRAINING PIPELINE")
    print(f"{'='*60}\n")
    
    # Check data file exists
    if not os.path.exists(data_file):
        print(f"❌ Error: {data_file} not found!")
        print("   Available options:")
        available_files = [f for f in os.listdir('.') if f.endswith('.txt') or f.endswith('.py')]
        for f in available_files:
            size = os.path.getsize(f)
            print(f"     - {f} ({size:,} bytes)")
        print("\n   Try running:")
        print("   python expand_data.py")
        return
    
    # Check file size and warn about memory
    file_size = os.path.getsize(data_file)
    file_size_mb = file_size / 1024 / 1024
    print(f"📊 Dataset info:")
    print(f"   File: {data_file}")
    print(f"   Size: {file_size_mb:.1f} MB")
    
    if file_size_mb > 100:
        print("   ⚠️  Large dataset detected!")
        print("   💡 Memory optimization enabled")
        print("   💡 Using chunked loading")
        if batch_size > 4:
            print(f"   💡 Reduced batch size to {batch_size}")
    
    # Step 1: Train tokenizer
    tokenizer = train_tokenizer_on_new_data()
    print()

    # Set device safely
    resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {resolved_device}")
    if resolved_device.type == "cpu":
        print("   ⚠️  CUDA not available, using CPU. Ensure venv312 with CUDA PyTorch is active.")

    # Step 2: Load dataset with memory optimization
    print("📊 Step 2: Loading and preparing dataset...")
    try:
        dataset = TextDataset(data_file, tokenizer, 
                            max_length=max_length, 
                            chunk_size=chunk_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True)  # Hardware acceleration
        print(f"   Data batches: {len(dataloader)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Sequence length: {max_length}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Try using a smaller dataset:")
        print("   python train_complete.py --data data.txt")
        return
    print()
    
    # Step 3: Create model
    print("🏗️  Step 3: Building model...")
    model = MiniGPT(
        vocab_size=30000,
        embed_size=256,
        heads=8,
        layers=4
    ).to(resolved_device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Device: {resolved_device}")
    
    # Memory info
    if resolved_device.type == "cuda":
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
        except Exception:
            pass
    print()
    
    # Step 4: Setup training
    print("⚙️  Step 4: Setting up training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    # Training directory
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Resume Logic
    start_epoch = 0
    import re
    checkpoint_pattern = re.compile(rf"^{model_name}_epoch(\d+)\.pt$")
    latest_epoch = -1
    latest_checkpoint = None
    
    for filename in os.listdir("checkpoints"):
        match = checkpoint_pattern.match(filename)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_checkpoint = os.path.join("checkpoints", filename)
                
    if latest_checkpoint:
        print(f"🔄 Found previous checkpoint: {latest_checkpoint}")
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=resolved_device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"   Successfully loaded state! Resuming from epoch {start_epoch+1}...")
        except Exception as e:
            print(f"   ⚠️ Failed to load checkpoint: {e}")
            print("   Starting from scratch instead.")
    else:
        print("🆕 No previous checkpoint found. Starting from scratch.")
    
    training_log = {
        "model_name": model_name,
        "data_file": data_file,
        "file_size_mb": file_size_mb,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "chunk_size": chunk_size,
        "device": str(resolved_device),
        "total_parameters": total_params,
        "start_time": datetime.now().isoformat(),
        "epochs": []
    }
    print()
    
    # Step 5: Training loop
    print("🔥 Step 5: Training model...\n")
    model.train()
    
    if start_epoch >= num_epochs:
        print(f"✅ Model has already been trained for {start_epoch} epochs.")
        print(f"   To train further, increase the number of epochs (currently {num_epochs}).")
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(resolved_device, non_blocking=True)
            target_ids = target_ids.to(resolved_device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids)
            
            # Calculate loss
            loss = criterion(
                logits.reshape(-1, 30000),
                target_ids.reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        print(f"✓ Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/{model_name}_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"   Checkpoint saved: {checkpoint_path}\n")
        
        training_log["epochs"].append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "checkpoint": checkpoint_path
        })
    
    # Save final model
    final_model_path = f"checkpoints/{model_name}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Training complete! Model saved: {final_model_path}")
    
    # Save training log
    log_path = f"logs/training_log_{model_name}.json"
    training_log["end_time"] = datetime.now().isoformat()
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"📋 Training log saved: {log_path}")
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MiniGPT model")
    parser.add_argument("--data", default="combined_training_data.txt", 
                       help="Dataset file to use")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help="Batch size (smaller = less memory)")
    parser.add_argument("--lr", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, 
                       help="Sequence length (smaller = less memory)")
    parser.add_argument("--chunk-size", type=int, default=500000, 
                       help="File chunk size in bytes")
    parser.add_argument("--name", default="minigpt", 
                       help="Model name for checkpoints")
    
    args = parser.parse_args()
    
    # Train the model
    model = train_model(
        model_name=args.name,
        data_file=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        chunk_size=args.chunk_size
    )
