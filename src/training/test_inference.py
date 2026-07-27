import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def run_inference():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = "models/financial_qlora_adapter"

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="cpu"
    )

    print("Loading fine-tuned LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    sample_statement = (
        "Operating profit for the fourth quarter increased by 18% "
        "year-over-year to $450 million, driven by strong cloud segment demand."
    )
    instruction = "Analyze the financial statement and determine the market sentiment."

    prompt = (
        f"<|im_start|>system\nYou are a financial analyst AI.<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n\nStatement: {sample_statement}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("Generating response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )

    print("\n" + "=" * 50)
    print("FINANCIAL STATEMENT:", sample_statement)
    print("-" * 50)
    print("MODEL RESPONSE:", response)
    print("=" * 50)


if __name__ == "__main__":
    run_inference()