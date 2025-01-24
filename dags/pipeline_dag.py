from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from main import data_ingestion, data_preprocessing, train, evaluate, deploy_to_HF

# Define args for the DAG
default_args = {
    'owner': 'hezron',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'fine_tune_pipeline',
    default_args=default_args,
    description='Pipeline for fine-tuning and deploying a Mental-Health-ChatBot',
    schedule_interval=None,  # Set to None to run manually
    start_date=datetime(2025, 1, 20),
    catchup=False,
    tags=['fine-tuning'],
) as dag:

    # Task 1: Data Ingestion
    task_data_ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=data_ingestion
    )

    # Task 2: Data Preprocessing
    def preprocessing_task(**kwargs):
        train_dataset, val_dataset, test_dataset, tokenizer = data_preprocessing()
        # Push values to XCom for downstream tasks
        kwargs['ti'].xcom_push(key='train_dataset', value=train_dataset)
        kwargs['ti'].xcom_push(key='val_dataset', value=val_dataset)
        kwargs['ti'].xcom_push(key='test_dataset', value=test_dataset)
        kwargs['ti'].xcom_push(key='tokenizer', value=tokenizer)

    task_data_preprocessing = PythonOperator(
        task_id='data_preprocessing',
        python_callable=preprocessing_task,
        provide_context=True
    )

    # Task 3: Training
    def training_task(**kwargs):
        # Retrieve datasets and tokenizer from XCom
        ti = kwargs['ti']
        train_dataset = ti.xcom_pull(task_ids='data_preprocessing', key='train_dataset')
        val_dataset = ti.xcom_pull(task_ids='data_preprocessing', key='val_dataset')
        tokenizer = ti.xcom_pull(task_ids='data_preprocessing', key='tokenizer')

        # Paths for saving
        bucket_name = "fine-tuning-mental-health-qwen"
        bucket_path = "model/model_weights"
        model_save_path = f"C:/Users/Hezron Ling/Desktop/Fine Tuning GCP/fine_tune_model/final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Train the model
        ft_model = train(train_dataset, val_dataset, tokenizer, bucket_name, bucket_path, model_save_path)
        
        # Push the fine-tuned model to XCom for downstream tasks
        ti.xcom_push(key='ft_model', value=ft_model)
        ti.xcom_push(key='model_save_path', value=model_save_path)

    task_training = PythonOperator(
        task_id='train_model',
        python_callable=training_task,
        provide_context=True
    )

    # Task 4: Evaluation
    def evaluation_task(**kwargs):
        # Retrieve fine-tuned model, tokenizer, and test dataset from XCom
        ti = kwargs['ti']
        ft_model = ti.xcom_pull(task_ids='train_model', key='ft_model')
        test_dataset = ti.xcom_pull(task_ids='data_preprocessing', key='test_dataset')
        tokenizer = ti.xcom_pull(task_ids='data_preprocessing', key='tokenizer')

        # Evaluate the model
        rouge_scores = evaluate(ft_model, tokenizer, test_dataset)
        
        # Log evaluation results
        print(f"ROUGE scores: {rouge_scores}")

        # Push evaluation results to XCom (optional, if needed for downstream use)
        ti.xcom_push(key='rouge_scores', value=rouge_scores)

    task_evaluation = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluation_task,
        provide_context=True
    )

    # Task 5: Deployment
    def deploy_task(**kwargs):
        # Retrieve the fine-tuned model and tokenizer from XCom
        ti = kwargs['ti']
        ft_model = ti.xcom_pull(task_ids='train_model', key='ft_model')
        model_save_path = ti.xcom_pull(task_ids='train_model', key='model_save_path')
        tokenizer = ti.xcom_pull(task_ids='data_preprocessing', key='tokenizer')

        # Deploy the model to HuggingFace
        deploy_to_HF(ft_model, datetime.now().strftime("%Y%m%d_%H%M%S"), tokenizer, model_save_path)

    task_deploy = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_task,
        provide_context=True
    )

    # Define task dependencies
    task_data_ingestion >> task_data_preprocessing >> task_training >> task_evaluation >> task_deploy
