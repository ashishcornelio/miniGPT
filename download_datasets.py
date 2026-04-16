"""
Download maximum possible Python datasets from Hugging Face by accumulating
non-gated instructions, ensuring a gigantic corpus 
"""
from datasets import load_dataset
import os

def extract_code_from_dataset(dataset):
    code_snippets = []
    for item in dataset:
        if isinstance(item, dict):
            # Check all possible columns/keys that hold python code data
            code = (item.get("output") or 
                   item.get("solution") or 
                   item.get("code") or 
                   item.get("answer") or 
                   item.get("content") or "")
            if code and len(code) > 20:
                code_snippets.append(code)
    return code_snippets

def save_to_file(code_snippets, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        for snippet in code_snippets:
            # We strictly append the raw code chunks followed by clear delimiters
            f.write(snippet)
            f.write("\n\n" + "="*50 + "\n\n")
    print(f"Saved {len(code_snippets)} python snippets to dataset cache.")

if __name__ == "__main__":
    output_file = "python_training_data.txt"
    if os.path.exists(output_file):
        os.remove(output_file)
        
    print("Accumulating maximum possible Python datasets from Hugging Face...\n")
    
    # A massive curated list of the largest open-source coding datasets on the entirety of Hugging Face
    datasets_to_pull = [
        ("flytech/python-codes-25k", "train"),
        ("iamtarun/python_code_instructions_18k_alpaca", "train"),
        ("m-a-p/CodeFeedback-Filtered-Instruction", "train"),
        ("TokenBender/code_instructions_122k_alpaca_style", "train"),
        ("nickrosh/Evol-Instruct-Code-80k-v1", "train"),
        ("sahil2801/CodeAlpaca-20k", "train"),
        ("nampdn-ai/tiny-codes", "train[:50000]") # Taking 50k from TinyCodes
    ]
    
    total_downloaded = 0
    
    for repo, split in datasets_to_pull:
        try:
            print(f"Downloading {repo}...")
            ds = load_dataset(repo, split=split)
            snippets = extract_code_from_dataset(ds)
            save_to_file(snippets, output_file)
            total_downloaded += len(snippets)
            print(f"✅ Successfully appended {repo}!\n")
        except Exception as e:
            print(f"❌ Skipping {repo} due to native load error: {e}\n")

    print(f"🔥 Download sequence finished! Agreggated {total_downloaded} full code samples across multiple repositories.")
    
    if os.path.exists(output_file):
        final_size = os.path.getsize(output_file) / (1024*1024)
        print(f"Final Extracted Text Data Size: {final_size:.1f} MB")
