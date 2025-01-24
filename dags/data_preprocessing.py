from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
)
from huggingface_hub import login
from google.cloud import bigquery
from datetime import datetime
from functools import partial

def fetch_dataset(key_path, project_id, dataset_id, table_id):
    # Initialize BigQuery
    client = bigquery.Client.from_service_account_json(key_path)

    # Get today's date
    today_date = datetime.now().strftime('%Y-%m-%d')

    # Query to fetch rows uploaded today
    query = f"""
    SELECT *
    FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE DATE(timestamp) = '{today_date}'
    """

    # Run the query and convert to DataFrame
    query_job = client.query(query)
    df = query_job.to_dataframe()

    # Drop the 'timestamp' column
    df = df.drop(columns=['timestamp'])

    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    return dataset

def load_tokenizer(base_model_name):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name,
                                              trust_remote_code=True,
                                              padding_side="left",
                                              add_eos_token=True,
                                              add_bos_token=True,
                                              use_fast=False)

    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def create_prompt_formats(sample):
    # Define the instruction
    instruction = "Below is a conversation where the user shares a personal experience or feeling, and the response provides guidance, empathy, and actionable advice to support the user's mental health."

    # Format the question and answer into the desired prompt
    ques_prompt = f"<USER_CONTEXT>: {sample['Context']}"
    ans_response = f"<RESPONSE>: {sample['Response']}"

    # Combine the instruction, question, and answer into a single text
    formatted_prompt = f"{instruction}\n{ques_prompt}\n{ans_response}"

    # Add to the new 'text' column
    sample["text"] = formatted_prompt

    return sample

def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

# To process the entire dataset
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset):
    # Add prompt to each sample
    dataset = dataset.map(create_prompt_formats)#, batched=True)

    # Apply preprocessing to each batch of the dataset & and remove existing fields 'Context' and 'Response'
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['Context', 'Response'],
    )

    # Filter out rows with input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset



