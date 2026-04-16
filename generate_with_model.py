"""
Load a trained model and generate predictions
"""

import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer
import os

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

def load_model_and_tokenizer(model_path, device="cpu"):
    """Load trained model and tokenizer"""
    
    # Load tokenizer
    if not os.path.exists("tokenizer/vocab.json"):
        print("❌ Error: Tokenizer not found!")
        print("   Run: python train_complete.py")
        return None, None
    
    tokenizer = ByteLevelBPETokenizer(
        vocab="tokenizer/vocab.json",
        merges="tokenizer/merges.txt"
    )
    
    # Load model
    model = MiniGPT(
        vocab_size=30000,
        embed_size=256,
        heads=8,
        layers=4
    ).to(device)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("   Available checkpoints:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                print(f"     - {f}")
        return None, None
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"✅ Model loaded: {model_path}")
    print(f"✅ Tokenizer loaded")
    
    return model, tokenizer

def generate_code(model, tokenizer, prompt, max_length=100, temperature=0.7, device="cpu"):
    """Generate Python code based on prompt"""
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    print(f"\n📝 Prompt: {prompt}")
    print(f"🔄 Generating code...\n")
    
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length - len(tokens)):
            # Get prediction for next token
            outputs = model(input_ids)
            logits = outputs[0, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            input_ids = torch.tensor([generated], dtype=torch.long).to(device)
            
            # Stop if we generate end token
            if next_token == tokenizer.token_to_id("[EOS]") or next_token == 0:
                break
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated)
    
    print("Generated Code:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    return generated_text

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate code with MiniGPT")
    parser.add_argument("--model", default="checkpoints/minigpt_final.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--prompt", default="def ",
                       help="Starting prompt")
    parser.add_argument("--length", type=int, default=100,
                       help="Max generated tokens")
    parser.add_argument("--temp", type=float, default=0.7,
                       help="Temperature (higher = more random)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}\n")
    
    model, tokenizer = load_model_and_tokenizer(args.model, device=device)
    
    if model and tokenizer:
        generate_code(model, tokenizer, args.prompt, 
                     max_length=args.length, 
                     temperature=args.temp,
                     device=device)
