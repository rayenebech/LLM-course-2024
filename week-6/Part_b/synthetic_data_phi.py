import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    device_map="cuda", 
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")


def load_queries(csv_file_path):
    """Load queries from a CSV file and return them as a list."""
    df = pd.read_csv(csv_file_path)
    return df['Query'].tolist()

def extract_abbreviations(queries):
    """Extract known abbreviations from the queries."""
    import re
    abbreviations = set()
    for query in queries:
        abbreviations.update(re.findall(r'\b[A-Z]{2,}\b', query))
    return abbreviations

def extract_first_json_block(response):
    """Extract the first JSON block from the response."""
    try:
        start = response.index("[")
        end = response.index("]", start) + 1
        json_block = response[start:end]
        return json.loads(json_block)
    except (ValueError, json.JSONDecodeError):
        return []

def generate_misspellings_with_model(query, abbreviations, N):
    """Generate N misspellings for a query using the LLM."""
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    prompt = (
        f"You are tasked with generating {N} misspellings for the following query. "
        f"Make the misspellings varied (phonetic, omission, transposition, repetition). "
        f"Avoid changing known abbreviations: {', '.join(abbreviations)}. "
        f"Do not provide any explanation. The output should be formatted as a JSON list of {N} sentences with different misspellings. "
        f"Query: \"{query}\". Provide a list of {N} misspellings."
    )

    generation_args = {
        "max_new_tokens": 200,
        "temperature": 0.7,
        "return_full_text": False,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id
    }

    output = pipe(prompt, **generation_args)
    response = output[0]['generated_text']

    misspellings = extract_first_json_block(response)
    return misspellings

def generate_misspellings(queries, abbreviations, N):
    """Generate misspellings for all queries."""
    misspelled_queries = {}

    for query in queries:
        misspelled_queries[query] = generate_misspellings_with_model(query, abbreviations, N)

    return misspelled_queries

def save_to_json(data, output_file):
    """Save the generated misspelled queries to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)



if __name__ == "__main__":
    csv_file_path = "web_search_queries.csv"  
    output_file = "misspelled_queries.json"
    queries = load_queries(csv_file_path)
    abbreviations = extract_abbreviations(queries)
    N = int(input("Enter the number of misspellings to generate per query: "))
    misspelled_queries = generate_misspellings(queries, abbreviations, N)
    save_to_json(misspelled_queries, output_file)
    print(f"Misspelled queries saved to {output_file}")