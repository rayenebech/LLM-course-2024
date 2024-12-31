from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch

from dotenv import load_dotenv
from threading import Thread
import os

from utils.helpers import singleton

load_dotenv()

@singleton
class LLamaModel:
    def __init__(self, **kwargs) -> None:
        self.model_name = kwargs.get("model_name")
        self.temperature = kwargs.get("temperature", 0.1)
        self.max_new_tokens = kwargs.get("max_tokens", 2048)
        self.stream = kwargs.get("stream", True)
        self.model_cache = kwargs.get("model_cache")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache)
        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                cache_dir=self.model_cache, 
                device_map="auto", 
                torch_dtype = torch.float16,
                quantization_config = {"load_in_4bit": True}
                )
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.generation_kwargs = dict(temperature=self.temperature, max_new_tokens=self.max_new_tokens)

        self.gen_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer = self.tokenizer,
            device_map="auto",
            streamer=self.streamer
        )
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(""),
        ]

    def generate(self, messages):
        generation_kwargs = dict(text_inputs = messages, return_full_text=False, **self.generation_kwargs)
        thread = Thread(target=self.gen_pipeline, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        for new_text in self.streamer:
            yield new_text
            generated_text += new_text    

if __name__ == "__main__":
    model = LLamaModel(model_name = "meta-llama/Llama-3.1-8B-Instruct")
    messages = [
    {"role": "system", "content": "You are a helpful finance assistant developed by Tietoevry."},
    {"role": "user", "content": "Who are you?"},
    ]
    gen = model.generate(messages)
    for token in gen:
        print(token)
