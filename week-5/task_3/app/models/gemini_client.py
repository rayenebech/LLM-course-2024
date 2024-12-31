import google.generativeai as genai

from dotenv import load_dotenv
import os

from utils.helpers import singleton, generator_simulator

load_dotenv()

GOOGLE_AI_KEY = os.getenv("GOOGLE_API_KEY")

@singleton
class GeminiClient:
    def __init__(self, **kwargs):
        genai.configure(api_key=GOOGLE_AI_KEY)
        self.model_name = kwargs.get("model_name")
        self.temperature = kwargs.get("temperature", 0)
        self.max_output_tokens = kwargs.get("max_output_tokens", 8000)
        self.stream = kwargs.get("stream", True)
        self.generation_kwargs = dict(temperature=self.temperature, max_output_tokens=self.max_output_tokens)
    
    def prepare_messages(self, messages):
        system_prompt = messages[0]["content"]
        new_messages = []
        for message in messages:
            if message["role"] == "user":
                new_messages.append({
                    "role": "user",
                    "parts": message["content"]
                })
            elif message["role"] == "assistant":
                new_messages.append({
                    "role": "model",
                    "parts": message["content"]
                })
        return system_prompt, new_messages
    

    def generate(self, messages):
        system_prompt, new_messages = self.prepare_messages(messages[:-1])
        self.model = genai.GenerativeModel(self.model_name, system_instruction = system_prompt)
        chat = self.model.start_chat(history=new_messages)
        response = chat.send_message(messages[-1]["content"], stream=self.stream)
        del self.model
        if self.stream:
            for chunk in response:
                for message in chunk.text:
                    yield message
        else:
            return generator_simulator(response.text)
        

