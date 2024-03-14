import torch
from transformers import BertForTokenClassification, BertTokenizer
from transformers import AutoModelForTokenClassification, Trainer, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, pipeline


label_list=["O", "B-HUSTYP", "B-KONSTRUKTIONSDETALJ", "B-LST_DNR", "B-SR_SYSTEM", "B-SR_KOORDINATER", "B-INTRASIS", "I-HUSTYP", "I-KONSTRUKTIONSDETALJ", "I-LST_DNR", "I-SR_SYSTEM", "I-SR_KOORDINATER", "I-INTRASIS"]
main_path = "/home/adam/code/bertbased/kbtraining/checkpoint-15000/"
# Step 1: Load pre-trained model and tokenizer
model_path = main_path + "model.safetensors"
tokenizer_path = main_path + "tokenizer.json"
print("model path: ", model_path)


model = AutoModelForTokenClassification.from_pretrained(main_path)
tokenizer = AutoTokenizer.from_pretrained(main_path)



# Step 2: Load additional files if required (config, optimizer, scheduler, etc.)
config_path = main_path + "config.json"
optimizer_path = main_path +"optimizer.pt"
scheduler_path = main_path + "scheduler.pt"

# Load config (if needed)
# config = BertConfig.from_json_file(config_path)

# Load optimizer and scheduler states (if needed)
# optimizer_state = torch.load(optimizer_path)
# scheduler_state = torch.load(scheduler_path)

# Step 3: Perform inference or testing
# Example input text
input_text = "Utredningen har utförts enligt beslut av Länsstyrelsen i Västra Götalands\nlän (dnr 220-39195-99) och har bekostats av Alvereds golf. Länsstyrelsens dnr: 220-39195-99. Koordinater för undersökningsytans sydvästra hörn:\nx 6395,00  y 1272,25y."
#input_text ="Böndernas hus."
# Tokenize input text
tokens = tokenizer.tokenize(input_text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

# Perform inference
with torch.no_grad():
    outputs = model(torch.tensor([input_ids]))

# Get predicted labels
predicted_labels = torch.argmax(outputs.logits, dim=2).squeeze()
print(predicted_labels)

# Decode labels
decoded_labels = [tokenizer.decode([label]) for label in predicted_labels]
res_labels = [label_list[label] for label in predicted_labels]

# Display results
print("Input Text:", input_text)
print("Predicted Labels:", decoded_labels)
print("Result", res_labels)
print(tokens)


nlp =  pipeline('ner', model = main_path, tokenizer=main_path)
print(nlp(input_text))
