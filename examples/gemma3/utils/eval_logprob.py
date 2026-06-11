import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="User: Write a short poem about the sky.\nModel: ")
    parser.add_argument("--target", type=str, default="The sky is blue,\nAnd so are you.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    
    # We want to measure the logprob of the target given the prompt
    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    target_ids = tokenizer(args.target, return_tensors="pt", add_special_tokens=False).input_ids
    
    input_ids = torch.cat([prompt_ids, target_ids], dim=-1).to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    
    P = prompt_ids.shape[1]
    T = target_ids.shape[1]
    
    target_log_probs = []
    for i in range(T):
        label_idx = P - 1 + i
        token_id = shift_labels[0, label_idx]
        token_log_prob = log_probs[0, label_idx, token_id].item()
        target_log_probs.append(token_log_prob)
        
    avg_log_prob = sum(target_log_probs) / len(target_log_probs)
    print(f"Model: {args.model_path}")
    print(f"Average target log probability: {avg_log_prob:.4f}")
    
if __name__ == "__main__":
    main()
