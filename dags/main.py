from datasets import DatasetDict, load_dataset
from datetime import datetime
from huggingface_hub import interpreter_login
from huggingface_hub import login
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_from_disk
from google.cloud import storage
from google.cloud import bigquery   
from datetime import datetime
from functools import partial
import os
from dotenv import load_dotenv

from data_upload import process_data, upload_df_to_bigquery
from data_preprocessing import fetch_dataset, load_tokenizer, preprocess_dataset
from train import load_quantized_model, initialize_peft_trainer, upload_model_to_gcp, merge_model, evaluate_rouge_scores, push_model

# Login to Huggingface
load_dotenv()
hf_api = os.getenv("HUGGINGFACE_API_KEY")
login(token=hf_api)

# Login to W&B
wandb.login()

# Login GCP
key_path = "C:/Users/Hezron Ling/Desktop/keysfine-tuning-llm-448515-731384ffa31b.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

# Initialize BigQuery
project_id = "fine-tuning-llm-448515"
dataset_id = "mental_health_data"
table_id = "conversations"

# Initialize bucket
bucket_name = "fine-tuning-mental-health-qwen"
bucket_path = "model/model_weights"

# Etc
max_length = 32768
base_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
seed = 42
time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Local paths
model_save_path = f"C:/Users/Hezron Ling/Desktop/Fine Tuning GCP/fine_tune_model/final_model_{time}"
cpoint_save_path = f"C:/Users/Hezron Ling/Desktop/Fine Tuning GCP/fine_tune_model/checkpoint_model_{time}"

def data_ingestion(): 
    # Load the dataset
    dataset = load_dataset("Amod/mental_health_counseling_conversations")

    # Apply the cleaning function
    processed_df = process_data(dataset)
    
    # Upload to BigQuery
    upload_df_to_bigquery(processed_df, key_path, project_id, dataset_id, table_id)

from datasets import DatasetDict

def data_preprocessing():
    # Fetch dataset
    dataset_today = fetch_dataset(key_path, project_id, dataset_id, table_id)

    # Load tokenizer
    tokenizer = load_tokenizer(base_model_name)

    # Split dataset: 80% train, 20% temp (to further split into validation and test)
    train = dataset_today.train_test_split(test_size=0.2, seed=42)

    # Further split the temp dataset into validation (50%) and test (50%) of the remaining 20%
    val_test = train['test'].train_test_split(test_size=0.5, seed=42)

    # Combine into a DatasetDict
    dataset = DatasetDict({
        'train': train['train'],        # 80%
        'validation': val_test['train'],  # 10%
        'test': val_test['test']          # 10%
    })

    test_dataset = dataset["test"]

    # Preprocess the train, validation, and test datasets
    train_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset['train'])
    val_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset['validation'])

    return train_dataset, val_dataset, test_dataset, tokenizer


def train(train_dataset, val_dataset, tokenizer, bucket_name, bucket_path, model_save_path):
    # Load quantized model
    base_model = load_quantized_model(base_model_name)  

    # Initialize peft_trainer
    peft_trainer = initialize_peft_trainer(train_dataset, val_dataset, cpoint_save_path, base_model, tokenizer)

    # Start training the model
    peft_trainer.train()    

    # Stop reporting to wandb
    wandb.finish()

    # Save model
    peft_trainer.save_model(model_save_path)

    # Upload adapter to GCP
    upload_model_to_gcp(model_save_path, bucket_name, bucket_path, time) 

    ft_model = merge_model(base_model, model_save_path)

    return ft_model

def evaluate(ft_model, tokenizer, test_dataset):
    # Evaluate ROUGE scores
    rouge_scores = evaluate_rouge_scores(ft_model, tokenizer, test_dataset)

    return rouge_scores

def deploy_to_HF(base_model, time, tokenizer, model_save_path):
    # Define the name 
    pushed_model_name = f"Qwen-0.5B-Mental-Health-{time}"

    # Push the model to HuggingFace
    push_model(base_model, pushed_model_name, tokenizer, model_save_path)


