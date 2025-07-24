
# models/custom_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class CustomModelAPI:
    def __init__(self, model_path="../models/finetuned-code-model"):
        """Initialize the custom model API"""
        print(f"Loading model from {model_path}...")

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_code(self, prompt, max_new_tokens=1024, temperature=0.2):
        """Generate code using the custom fine-tuned model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate output
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode output
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Return only the newly generated part (excluding the input prompt)
            return generated_text[len(prompt):]

        except Exception as e:
            print(f"Error generating code with custom model: {e}")
            return ""

