#!/usr/bin/env python3
"""
Interactive Medical Q&A with Your Trained Model
Run: python test_model.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

print("🏥 Loading your trained medical Q&A model...")
print("   (This may take 10-20 seconds...)\n")

# Check if model exists
MODEL_PATH = "./final_model"
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    print("\n💡 You need to train the model first!")
    print("   1. Open Medical_QA_Training_Simplified.ipynb")
    print("   2. Run all cells (this trains and saves the model)")
    print("   3. Then run this script again\n")
    exit(1)

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
print(f"✅ Model loaded on {device}!\n")

def ask_question(question, temperature=0.3, max_length=200):
    """Generate answer to medical question"""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the answer part
    if "Answer:" in full_text:
        answer = full_text.split("Answer:")[1].strip()
    else:
        answer = full_text.strip()
    
    return answer

# Interactive loop
print("="*80)
print("           🏥 MEDICAL Q&A ASSISTANT - INTERACTIVE MODE")
print("="*80)
print("\n💡 Tips:")
print("   - Ask medical questions (e.g., 'What causes diabetes?')")
print("   - Type 'quit' or 'exit' to stop")
print("   - Type 'temp X' to change temperature (0.1-1.5)")
print("   - Lower temp = more focused, Higher temp = more creative\n")
print("="*80)

current_temp = 0.3
print(f"\n🌡️  Current temperature: {current_temp} (medical precision mode)")

while True:
    print("\n" + "─"*80)
    question = input("❓ Your Question: ").strip()
    
    if not question:
        continue
    
    if question.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Thanks for using the Medical Q&A Assistant! Stay healthy!")
        break
    
    # Temperature adjustment
    if question.lower().startswith('temp '):
        try:
            new_temp = float(question.split()[1])
            if 0.1 <= new_temp <= 1.5:
                current_temp = new_temp
                print(f"🌡️  Temperature set to: {current_temp}")
            else:
                print("⚠️  Temperature must be between 0.1 and 1.5")
        except:
            print("⚠️  Invalid temperature format. Use: temp 0.5")
        continue
    
    print("\n🤖 Model Answer:")
    print("─"*80)
    try:
        answer = ask_question(question, temperature=current_temp)
        print(answer)
    except Exception as e:
        print(f"⚠️  Error: {e}")
    print("─"*80)

print("\n✅ Session ended.\n")
