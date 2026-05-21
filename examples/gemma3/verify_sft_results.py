import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def verify_sft(model_path, base_model_id="google/gemma-3-1b-it"):
    print(f"--- Loading SFT model from {model_path} ---")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    
    medical_queries = [
        "What are the differential diagnoses for a patient with acute shortness of breath and chest pain?",
        "What initial tests should be ordered for suspected acute coronary syndrome?",
        "What are common physical findings in a patient with pulmonary embolism?",
        "What is the standard emergency management for acute chest pain?",
        "Which imaging modality is preferred to rule out pulmonary embolism?"
    ]
    
    expected_keywords = [
        ["myocardial infarction", "pulmonary embolism", "aortic dissection"],
        ["EKG", "troponin"],
        ["tachypnea", "tachycardia", "92%"],
        ["oxygen", "aspirin", "nitroglycerin"],
        ["CT pulmonary angiogram", "CTPA"]
    ]

    print("\n--- Running Memorization Test (Gold Samples) ---")
    pass_count = 0
    for i, query in enumerate(medical_queries):
        chat = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=50)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"Q: {query}")
        print(f"A: {response.strip()}")
        
        # Keyword based verification for robustness
        match = all(kw.lower() in response.lower() for kw in expected_keywords[i])
        if match:
            print("✓ PASS")
            pass_count += 1
        else:
            print("✗ FAIL")
        print("-" * 30)

    print(f"\nMemorization Score: {pass_count}/{len(medical_queries)}")

    print("\n--- Running Format & Style Test ---")
    test_query = "What is the capital of France?"
    chat = [{"role": "user", "content": test_query}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20, return_dict_in_generate=True, output_scores=True)
    
    decoded_full = tokenizer.decode(outputs.sequences[0])
    
    print(f"Raw output: {decoded_full}")
    
    # Check for correct chat template markers
    if "<start_of_turn>model" in decoded_full and "<end_of_turn>" in decoded_full:
        print("✓ Chat Template Alignment: PASS")
    else:
        print("✗ Chat Template Alignment: FAIL")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the exported HF model")
    args = parser.parse_args()
    
    verify_sft(args.model_path)
