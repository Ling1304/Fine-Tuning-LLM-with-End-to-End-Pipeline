from datasets import load_dataset
from huggingface_hub import interpreter_login
from huggingface_hub import login
from datetime import datetime
from google.cloud import bigquery

# Define the processing function
def process_data(dataset):
    # Convert the dataset to a pandas DataFrame
    df = dataset['train'].to_pandas()

    # Remove rows where any column contains 'www', 'http', 'https', '-', or '/'
    df = df[~df.apply(lambda row: row.astype(str).str.contains(r'www|http|https|-|/|~', regex=True).any(), axis=1)]

    # Remove duplicates in the 'Context' column
    df = df.drop_duplicates(subset='Context', keep='first').reset_index(drop=True)

    # Clean the 'Response' column by removing text after the last full stop (due to sign offs)
    def clean_response(text):
        return text[:text.rfind('.') + 1] if '.' in text else text

    df['Response'] = df['Response'].apply(clean_response)

    # Add a 'timestamp' column with the current time
    df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return df

def upload_df_to_bigquery(dataframe, key_path, project_id, dataset_id, table_id):
    # Initialize BigQuery client
    client = bigquery.Client.from_service_account_json(key_path)

    # Define the table reference
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    # Configure the BigQuery 
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,  # Add data to existing table
        autodetect=True,)

    # Upload the DataFrame to BigQuery
    job = client.load_table_from_dataframe(dataframe, table_ref, job_config=job_config)

    # Wait for the job to complete
    job.result()

    print(f"Data uploaded successfully to {table_ref}.")