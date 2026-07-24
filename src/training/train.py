# import os
# import torch
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
#     TrainingArguments
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from trl import SFTTrainer, SFTConfig


# class FinancialModelTrainer:
#     """Manages 4-bit QLoRA fine-tuning for domain-specific financial QA."""

#     def __init__(
#         self,
#         model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
#         dataset_path: str = "data/processed/financial_qa_train.jsonl",
#         output_dir: str = "models/financial_qlora_adapter"
#     ):
#         self.model_id = model_id
#         self.dataset_path = dataset_path
#         self.output_dir = output_dir

#     def format_prompt(self, sample: dict) -> dict:
#         """Formats dataset rows into standard instruction-tuning prompts."""
#         context = sample.get("context", "")
#         instruction = sample.get("instruction", "Analyze the financial statement.")
#         response = sample.get("response", "")

#         formatted_text = (
#             f"<|im_start|>system\nYou are a financial analyst AI.<|im_end|>\n"
#             f"<|im_start|>user\n{instruction}\n\nStatement: {context}<|im_end|>\n"
#             f"<|im_start|>assistant\n{response}<|im_end|>"
#         )
#         return {"text": formatted_text}

#     def train(self):
#         print(f"Loading dataset from {self.dataset_path}...")
#         raw_dataset = load_dataset("json", data_files=self.dataset_path, split="train")
#         formatted_dataset = raw_dataset.map(self.format_prompt)

#         print("Configuring 4-bit Quantization (bitsandbytes)...")
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True
#         )

#         print(f"Loading base model: {self.model_id}...")
#         tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token

#         model = AutoModelForCausalLM.from_pretrained(
#             self.model_id,
#             quantization_config=bnb_config,
#             device_map="auto",
#             trust_remote_code=True
#         )

#         # Prepare model for PEFT/k-bit training
#         model = prepare_model_for_kbit_training(model)

#         print("Setting up LoRA Adapter configuration...")
#         peft_config = LoraConfig(
#             r=16,
#             lora_alpha=32,
#             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#             lora_dropout=0.05,
#             bias="none",
#             task_type="CAUSAL_LM"
#         )

#         model = get_peft_model(model, peft_config)
#         model.print_trainable_parameters()

#         # Training configuration
#         sft_config = SFTConfig(
#             output_dir=self.output_dir,
#             dataset_text_field="text",
#             max_seq_length=512,
#             num_train_epochs=3,
#             per_device_train_batch_size=2,
#             gradient_accumulation_steps=4,
#             learning_rate=2e-4,
#             fp16=True,
#             logging_steps=10,
#             save_strategy="epoch",
#             warmup_ratio=0.03,
#             report_to="none"
#         )

#         trainer = SFTTrainer(
#             model=model,
#             train_dataset=formatted_dataset,
#             peft_config=peft_config,
#             processing_class=tokenizer,
#             args=sft_config
#         )

#         print("Starting QLoRA Fine-Tuning...")
#         trainer.train()

#         print(f"Saving trained adapter weights to {self.output_dir}...")
#         trainer.model.save_pretrained(self.output_dir)
#         tokenizer.save_pretrained(self.output_dir)
#         print("Fine-tuning completed successfully!")


# if __name__ == "__main__":
#     trainer = FinancialModelTrainer()
#     trainer.train()

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model


class FinancialModelTrainer:
    """CPU-Compatible LoRA Fine-Tuning Script using Hugging Face Trainer."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        dataset_path: str = "data/processed/financial_qa_train.jsonl",
        output_dir: str = "models/financial_qlora_adapter"
    ):
        self.model_id = model_id
        self.dataset_path = dataset_path
        self.output_dir = output_dir

    def format_and_tokenize(self, sample: dict, tokenizer, max_length: int = 256) -> dict:
        context = sample.get("context", "")
        instruction = sample.get("instruction", "Analyze the financial statement.")
        response = sample.get("response", "")

        formatted_text = (
            f"<|im_start|>system\nYou are a financial analyst AI.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}\n\nStatement: {context}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )

        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_length,
            padding=False
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def train(self):
        print(f"Loading dataset from {self.dataset_path}...")
        raw_dataset = load_dataset("json", data_files=self.dataset_path, split="train")

        print(f"Loading base model on CPU: {self.model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Tokenizing and formatting dataset...")
        tokenized_dataset = raw_dataset.map(
            lambda x: self.format_and_tokenize(x, tokenizer),
            remove_columns=raw_dataset.column_names
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cpu",
            trust_remote_code=True
        )

        print("Setting up LoRA Adapter configuration...")
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            use_cpu=True,
            logging_steps=5,
            save_strategy="no",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            args=training_args
        )

        print("Starting CPU LoRA Fine-Tuning...")
        trainer.train()

        print(f"Saving trained adapter weights to {self.output_dir}...")
        trainer.model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    trainer = FinancialModelTrainer()
    trainer.train()