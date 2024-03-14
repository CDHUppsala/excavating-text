
import json
import re
import os

# Function to split text into sentences with a maximum word count
def split_into_sentences(text, max_words=100):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    updated_sentences = []
    cumulative_offset = 0
    for sentence in sentences:
        words = sentence.split()
        if len(words) > max_words:
            for i in range(0, len(words), max_words):
                updated_sentence = ' '.join(words[i:i+max_words])
                updated_sentences.append((cumulative_offset, cumulative_offset + len(updated_sentence), updated_sentence))
                cumulative_offset += len(updated_sentence) + 1  # Add 1 for space
        else:
            updated_sentences.append((cumulative_offset, cumulative_offset + len(sentence), sentence))
            cumulative_offset += len(sentence) + 1  # Add 1 for space
    return updated_sentences

# Function to extract sentences containing the entities
def extract_sentence_for_annotation(text, annotation):
    start, end, _ = annotation
    sentences = split_into_sentences(text)
    for sentence_start, sentence_end, sentence in sentences:
        if start >= sentence_start and end <= sentence_end:
            return sentence, start - sentence_start, end - sentence_start
    return None, None, None

# Loop through JSON files in a directory
json_data = []
directory = "./"
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            for annotation in data["annotations"]:
                text = annotation[0]
                entities = annotation[1]["entities"]
                for entity in entities:
                    # Extract sentence containing the entity
                    sentence, start_offset, end_offset = extract_sentence_for_annotation(text, entity)
                    if sentence is not None:
                        # Save annotation separately with its text and updated positions
                        updated_text = sentence
                        updated_annotation = [start_offset, end_offset, entity[2]]  # Update start and end positions
                        print("original annotation", entities)
                        print("updated annotation", updated_annotation)
                        print("text length", len(text), len(updated_text))
                        print("==" * 20)                  

                        json_data.append({"text": updated_text, "annotation": updated_annotation})

# Save the updated data to separate JSON files
output_file = f"updated_data.json"
with open(output_file, "w", encoding="utf-8") as out_file:
    json.dump(json_data, out_file, ensure_ascii=False, indent=4)
