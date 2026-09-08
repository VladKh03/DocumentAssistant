from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class QwenLLM:
    def __init__(self, model_name="Qwen/Qwen3-4B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def build_prompt(self, question, chunks):
        context_parts = []

        for chunk in chunks:
            page = chunk["page"]
            text = chunk["text"]

            context_parts.append(
                f"[Page {page}]\n{text}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a document assistant.

Answer the user's question using only the context below.

Rules:
- Do not explain your reasoning
- Do not output analysis
- Do not output <think> tags
- Give a direct and concise answer
- If several items are mentioned, list them
- If the answer is not present in the context, say:
  "The information was not found in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

        return prompt

    def generate(self, question, chunks, max_new_tokens=256):
        prompt = self.build_prompt(
            question,
            chunks
        )

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return answer