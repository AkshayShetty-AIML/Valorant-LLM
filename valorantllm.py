!pip install torch transformers sentencepiece accelerate datasets peft bitsandbytes pytorch

!pip install -U bitsandbytes

!pip install datasets

from huggingface_hub import notebook_login

notebook_login()

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model

MODEL_NAME = "microsoft/phi-2"

#Model Names
#"meta-llama/Llama-2-7b-chat-hf"
#"mistralai/Mistral-7B-Instruct-v0.1"
#"microsoft/phi-2"
#"mistralai/Mistral-7B-v0.1"
#"meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True, device_map="auto")

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# Test the chatbot
def ask_chatbot(prompt):
    response = chatbot(prompt, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]

print(ask_chatbot("What are the best weapons in Elden Ring?"))

import json

with open("/content/ValorantDataset.json", "r", encoding="utf-8") as file:
    dataset = json.load(file)

def format_data(example):
    return {
        "text": f"User: {example['prompt']}\nBot: {example['response']}"
    }

formatted_dataset = [format_data(example) for example in dataset]

with open("formatted_ValorantDataset.json", "w", encoding="utf-8") as file:
    json.dump(formatted_dataset, file, indent=4)

dataset = load_dataset("json", data_files={"train": "/content/formatted_ValorantDataset.json"})

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)


lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none"
)

model = get_peft_model(model, lora_config)


training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()


model.save_pretrained("./fine_tuned_game_chatbot")

model.save_pretrained("./llama_valorant_finetuned")
tokenizer.save_pretrained("./llama_valorant_finetuned")

model.push_to_hub('ShettyAkshay/ValorantLLM_Phi2')



from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model

M_NAME = 'ShettyAkshay/ValorantLLM_Phi2'

tokenizer = AutoTokenizer.from_pretrained(M_NAME)
#quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(M_NAME, load_in_4bit=True, device_map="auto")

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

def ask_chatbot(prompt):
    response = chatbot(prompt, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]

print(ask_chatbot("What is the best way to control spray with the Vandal?"))

