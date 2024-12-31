import pandas as pd
import random
import json
import re

def load_queries(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df['Query'].tolist()

def extract_abbreviations(queries):
    abbreviations = set()
    for query in queries:
        abbreviations.update(re.findall(r'\b[A-Z]{2,}\b', query))
    return abbreviations

def generate_misspelling(query, abbreviations):
    words = query.split()
    misspelled_query = []
    for word in words:
        if word in abbreviations:
            misspelled_query.append(word)
            continue
        mutation_type = random.choice(["phonetic", "omission", "transposition", "repetition"])
        if mutation_type == "phonetic":
            misspelled_query.append(re.sub(r'ph', 'f', word, flags=re.IGNORECASE))
        elif mutation_type == "omission":
            if len(word) > 1:
                pos = random.randint(0, len(word) - 1)
                misspelled_query.append(word[:pos] + word[pos + 1:])
            else:
                misspelled_query.append(word)
        elif mutation_type == "transposition":
            if len(word) > 1:
                pos = random.randint(0, len(word) - 2)
                misspelled_query.append(word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:])
            else:
                misspelled_query.append(word)
        elif mutation_type == "repetition":
            pos = random.randint(0, len(word) - 1)
            misspelled_query.append(word[:pos] + word[pos] + word[pos] + word[pos + 1:])
        else:
            misspelled_query.append(word)
    return ' '.join(misspelled_query)

def generate_misspellings(queries, abbreviations, N):
    misspelled_queries = {}
    for query in queries:
        misspelled_variants = []
        for _ in range(N):
            misspelled_variants.append(generate_misspelling(query, abbreviations))
        misspelled_queries[query] = misspelled_variants
    return misspelled_queries

def save_to_json(data, output_file):
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
