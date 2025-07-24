from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Salesforce/codegen-350M-mono",
    use_auth_token="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"  # paste your actual token here as a string
)

# models/finetune_model.py
import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

def prepare_training_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    samples = [{"text": f"{item['prompt']}{item['completion']}"} for item in data]
    return Dataset.from_list(samples)

def finetune_model(
    base_model_name="Salesforce/codegen-350M-mono",
    data_path="C:\\Users\\annan\\code_generation_comparison\\data\\data\\python_dataset.json",
    output_dir="../models/finetuned-code-model",
    epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5
):
    print(f"Loading base model: {base_model_name}")
    
    # Step 1: Load the base model WITHOUT quantization first
    print("Loading model without quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Step 2: Extract model modules to properly configure LoRA
    model_modules = [name for name, _ in model.named_modules() if "proj" in name]
    print("Available modules:", model_modules)
    
    # Step 3: Configure the LoRA adapters
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["qkv_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Step 4: Apply PEFT/LoRA to the model
    model = get_peft_model(model, lora_config)
    print(f"PEFT model class: {model.__class__}")
    model.print_trainable_parameters()
    
    # Step 5: Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Step 6: Prepare the dataset
    print("Preparing dataset...")
    dataset = prepare_training_data(data_path)
    
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512),
        batched=True,
        remove_columns=["text"]
    )
    
    # Step 7: Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        fp16=True,
        optim="adamw_torch"
    )
    
    # Step 8: Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Step 9: Create and run the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    # Step 10: Save the model (only the LoRA adapter)
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

if __name__ == "__main__":
    finetune_model()