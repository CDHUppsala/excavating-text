# excavating-text

================

# Overview

---

This repository contains code for a natural language processing (NLP) project focused on text excavation. The project uses transformer-based models to perform named entity recognition (NER) on Swedish text data.

# Code Structure

---

The repository is organized into several files and directories:

- `experiments.ipynb`: A Jupyter notebook containing code for experimenting with different NLP models and techniques.
- `test.py`: A Python script for testing the NER model on a sample input text.
- `eval.py`: A Python script for evaluating the performance of the NER model on a labeled dataset.
- `extract-annotations.py`: A Python script for extracting annotations from a dataset.
- `kaggleupload.py`: A Python script for uploading the model to Kaggle.
- `kbtraining/checkpoint-15000/`: A directory containing model checkpoints and other training artifacts.

# Model

---

The project uses a transformer-based model for NER, specifically a `AutoModelForTokenClassification` model from the Hugging Face Transformers library. The model is trained on a Swedish dataset and is capable of recognizing various entity types, including:

- `LST_DNR`
- `SR_SYSTEM`
- `SR_KOORDINATER`
- `INTRASIS`
- `HUSTYP`
- `KONSTRUKTIONSDETALJ`

# Usage

---

To use the model, follow these steps:

1. Install the required dependencies, including the Hugging Face Transformers library.
2. Download the pre-trained model checkpoint from the `kbtraining/checkpoint-15000/` directory.
3. Load the model and tokenizer using the `AutoModelForTokenClassification` and `AutoTokenizer` classes.
4. Preprocess the input text using the `tokenizer` object.
5. Perform inference using the `model` object.
6. Decode the predicted labels using the `label_list` variable.

# Example

---

Here's an example of how to use the model:

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load pre-trained model and tokenizer
model_path = "./kbtraining/checkpoint-15000/model.safetensors"
tokenizer_path = "./kbtraining/checkpoint-15000/tokenizer.json"
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Preprocess input text
input_text = "Utredningen har utförts enligt beslut av Länsstyrelsen i Västra Götalands län (dnr 220-39195-99) och har bekostats av Alvereds golf."
tokens = tokenizer.tokenize(input_text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)

# Perform inference
with torch.no_grad():
	outputs = model(torch.tensor([input_ids]))

# Decode predicted labels
predicted_labels = torch.argmax(outputs.logits, dim=2).squeeze()
decoded_labels = [label_list[label] for label in predicted_labels]
print(decoded_labels)
```
