from fasthtml.common import *
import google.generativeai as genai
from google.generativeai import GenerationConfig
import strip_markdown
import configparser
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
LLM = "gemini-1.5-flash"
generation_config = GenerationConfig(temperature= 0) # Set temperature to 0 for deterministic output
model = genai.GenerativeModel(LLM, generation_config = generation_config)
chat = model.start_chat()
# Read system prompts from config file
prompts = configparser.ConfigParser()
prompts.read('prompts.env')

# Set system prompt
system_prompt = prompts.get("SYSTEM_PROMPTS", "BASIC_PROMPT")
role = prompts.get("TEMPLATE", "ROLE")
goal = prompts.get("TEMPLATE", "GOAL")
tone = prompts.get("TEMPLATE", "TONE")
few_shots = []
# Set up the app, including daisyui and tailwind for the chat component
hdrs = (picolink, Script(src="https://cdn.tailwindcss.com"),
    Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css"))
app = FastHTML(hdrs=hdrs, cls="p-4 max-w-lg mx-auto")

# Chat message component (renders a chat bubble)
def ChatMessage(msg, user):
    bubble_class = "chat-bubble-primary" if user else 'chat-bubble-secondary'
    chat_class = "chat-end" if user else 'chat-start'
    return Div(cls=f"chat {chat_class}")(
               Div('user' if user else 'assistant', cls="chat-header"),
               Div(msg, cls=f"chat-bubble {bubble_class}"),
               Hidden(msg, name="messages")
           )

# The input field for the user message.
def ChatInput():
    return Input(name='msg', id='msg-input', placeholder="Type a message",
                 cls="input input-bordered w-full", hx_swap_oob='true')

# The main screen
@app.get
def index():
    sidebar = Div(cls="sidebar w-64 p-4 bg-base-200 h-screen fixed left-0")(
        Form(hx_post="update_prompt", hx_target="#system-prompt")(
            Div(cls="form-control")(
                Label("Select Prompt Type:", cls="label"),
                Label(cls="label cursor-pointer")(
                    Span("Basic Prompt"),
                    Input(type="radio", name="prompt_type", value="basic", 
                          cls="radio", checked=True)
                ),
                Label(cls="label cursor-pointer")(
                    Span("Zero-shot Prompt"),
                    Input(type="radio", name="prompt_type", value="zero", 
                          cls="radio")
                ),
                Label(cls="label cursor-pointer")(
                    Span("One-shot Prompt"),
                    Input(type="radio", name="prompt_type", value="one", 
                          cls="radio")
                ),
                Label(cls="label cursor-pointer")(
                    Span("Two-shot Prompt"),
                    Input(type="radio", name="prompt_type", value="two", 
                          cls="radio")
                ),
                Label(cls="label cursor-pointer")(
                    Span("COT Prompt"),
                    Input(type="radio", name="prompt_type", value="cot", 
                          cls="radio")
                ),

            ),
            Button("Update Prompt", cls="btn btn-primary mt-4"),
            Div(id="system-prompt", cls="text-sm mt-4 p-2 bg-base-300 rounded")(
                f"Current prompt: {system_prompt}"
            )
        )
    )

    chat_area = Div()(
        Form(hx_post=send, hx_target="#chatlist", hx_swap="beforeend")(
            Div(id="chatlist", cls="chat-box h-[73vh] overflow-y-auto"),
            Div(cls="flex space-x-2 mt-2")(
                Group(ChatInput(), Button("Send", cls="btn btn-primary"))
            )
        )
    )
    
    return Titled('Simple chatbot demo', Div(sidebar, chat_area))

@app.post
def update_prompt(prompt_type: str):
    global system_prompt
    global few_shots 
    if prompt_type == "basic":
        system_prompt = prompts.get("SYSTEM_PROMPTS", "BASIC_PROMPT")
    elif prompt_type == "zero":
        system_prompt =  f"You are {role}. {goal}. {tone}" #prompts.get("SYSTEM_PROMPTS", "ZERO_SHOT_PROMPT")
        
    elif prompt_type == "one":
        system_prompt = f"You are {role}. {goal}. {tone}"
        few_shots = [prompts.get("EXAMPLES", "ONE_SHOT_QUESTION"), 
                          prompts.get("EXAMPLES", "ONE_SHOT_RESPONSE")]
    elif prompt_type == "two":
        system_prompt =  f"You are {role}. {goal}. {tone}"
        few_shots = [prompts.get("EXAMPLES", "ONE_SHOT_QUESTION"), 
                          prompts.get("EXAMPLES", "ONE_SHOT_RESPONSE"),
                          prompts.get("EXAMPLES", "TWO_SHOT_QUESTION"),
                          prompts.get("EXAMPLES", "TWO_SHOT_RESPONSE")]
    elif prompt_type == "cot":
        system_prompt =  f"You are {role}. {goal}. {tone}"
        system_prompt += " " + prompts.get("SYSTEM_PROMPTS", "COT_PROMPT")
    
    return Div(f"Current prompt: {system_prompt}")

# Handle the form submission
@app.post
def send(msg:str, messages:list[str]=None):
    if not messages: messages = [system_prompt]
    if len(few_shots) > 0:
        msg = "Customer: "  + msg.rstrip() 
        for shot in few_shots:
            messages.append(shot)
    messages.append(msg)
    print(messages)
    r = chat.send_message(messages).text
    return (ChatMessage(msg, True),    # The user's message
            ChatMessage(strip_markdown.strip_markdown(r.rstrip()), False), # The chatbot's response
            ChatInput()) # And clear the input field

serve()

