import json

# Function to convert entities to BIO tagging scheme
def convert_entities_to_bio(text, entities):
    bio_labels = ['O'] * len(texti.split())  # Initialize with 'O' for each token
    c = 0
    if len(entities):
        print(entities)
        start, end, label = entities
        print(start)
        for i in range(start, end):
            if i == start:
                bio_labels[i] = 'B-' + label  # Beginning of entity
            else:
                bio_labels[i] = 'I-' + label  # Inside of entity
    return bio_labels

# Load data from JSON file
input_file = "updated_data.json"
output_file = "ner_data.json"

formatted_data = []

with open(input_file, "r") as f:
    data = json.load(f)
    for entry in data:
        text = entry["text"]
        entities = entry["annotation"]
        # Prepare data in the required format
        formatted_entry = {
            'text': text,
            'labels': convert_entities_to_bio(text, entities)
        }
        formatted_data.append(formatted_entry)

# Save formatted data to a new JSON file
with open(output_file, "w") as f:
    json.dump(formatted_data, f, indent=4)

