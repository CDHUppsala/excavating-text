from transformers import AutoModelForTokenClassification, Trainer, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, AutoConfig
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_metric, ClassLabel
import pandas as pd
import evaluate
import json

metric = load_metric("seqeval")


classmap = ClassLabel(num_classes=13, names=["O", "B-HUSTYP", "B-KONSTRUKTIONSDETALJ", "B-LST_DNR", "B-SR_SYSTEM", "B-SR_KOORDINATER", "B-INTRASIS", "I-HUSTYP", "I-KONSTRUKTIONSDETALJ", "I-LST_DNR", "I-SR_SYSTEM", "I-SR_KOORDINATER", "I-INTRASIS"])

label_list=["O", "B-HUSTYP", "B-KONSTRUKTIONSDETALJ", "B-LST_DNR", "B-SR_SYSTEM", "B-SR_KOORDINATER", "B-INTRASIS", "I-HUSTYP", "I-KONSTRUKTIONSDETALJ", "I-LST_DNR", "I-SR_SYSTEM", "I-SR_KOORDINATER", "I-INTRASIS"]
main_path = "./kbtraining/checkpoint-15000/"
# Step 1: Load pre-trained model and tokenizer
model_path = main_path + "model.safetensors"
tokenizer_path = main_path + "tokenizer.json"
print("model path: ", model_path)


model = AutoModelForTokenClassification.from_pretrained(main_path)
tokenizer = AutoTokenizer.from_pretrained(main_path)


with open("ner_data.json", "r") as f:
	jsdata = json.load(f)

train_sentences, eval_sentences = train_test_split(jsdata, test_size=0.3, random_state=24)

ds_train = Dataset.from_pandas(pd.DataFrame(data=train_sentences))
ds_eval = Dataset.from_pandas(pd.DataFrame(data=eval_sentences))

data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

max_seq_length = 512

# Assuming you have ds_train and ds_eval datasets

ds_train = ds_train.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=max_seq_length))
ds_eval = ds_eval.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=max_seq_length))



# Map labels to integers
ds_train = ds_train.map(lambda y: {"labels": classmap.str2int(y["annotation"])})
ds_eval = ds_eval.map(lambda y: {"labels": classmap.str2int(y["annotation"])})

ds_train = ds_train.filter(lambda example: len(example["text"]) <= max_seq_length)
ds_eval = ds_eval.filter(lambda example: len(example["text"]) <= max_seq_length)




def compute_metrics(p):
	predictions, labels = p
	predictions = predictions.argmax(axis=2)
	# Remove ignored index (special tokens)
	true_predictions = [
		[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
		for prediction, label in zip(predictions, labels)
	]
	true_labels = [
		[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
		for prediction, label in zip(predictions, labels)
	]
	results = metric.compute(predictions=true_predictions, references=true_labels)
	# print(results)
	# return {
	# 	"precision": results["overall_precision"],
	# 	"recall": results["overall_recall"],
	# 	"f1": results["overall_f1"],
	# 	"accuracy": results["overall_accuracy"],
	# }


batch_size= 32

training_args = TrainingArguments(
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	num_train_epochs=1000, 
	logging_dir='./logs',
	logging_steps=10,
	evaluation_strategy="epoch",
	save_strategy="steps",
	save_steps=5000,
	report_to="none",
	#learning_rate=1e-3,
	output_dir="./iobkbtraining"
)


# Initialize our Trainer
trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=ds_train,
	eval_dataset=ds_eval,
	data_collator=data_collator,
	tokenizer=tokenizer,
	compute_metrics=compute_metrics,
)


# Evaluate the model
eval_result = trainer.evaluate()

# Print the evaluation result
print(eval_result)

