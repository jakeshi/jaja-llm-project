from transformers import pipeline

class LLMModel:
    def __init__(self, model_name="gpt2"):
        self.generator = pipeline('text-generation', model=model_name)

    def generate_text(self, prompt, max_length=50):
        return self.generator(prompt, max_length=max_length, num_return_sequences=1)[0]['generated_text']

if __name__ == "__main__":
    model = LLMModel()
    prompt = "Once upon a time"
    generated_text = model.generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")
