from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
import transformers
import torch
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from google.cloud import storage
from rouge_score import rouge_scorer

def load_quantized_model(base_model_name):
    # Ensure the computation uses 16-bit floating-point (reduce memory usage, speed up training)
    compute_dtype = getattr(torch, "float16")

    # Configure Bits and Bytes to load the model in 4-bit (quantized)
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, # Load the weights in 4 bit
            bnb_4bit_quant_type='nf4', # Use nf4 datatype
            bnb_4bit_compute_dtype=compute_dtype, # Uses 16-bit floating-point (float16)
            bnb_4bit_use_double_quant=True, # Enable double quantization
        )

    # Load the pretrained model
    device_map = "auto"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                          device_map=device_map,
                                                          quantization_config=bnb_config, # To load in 4-bit and double quantization
                                                          trust_remote_code=True,
                                                          use_cache = False,
                                                          use_auth_token=True)

    return base_model

def initialize_peft_trainer(train_dataset, val_dataset, output_dir, base_model, tokenizer):
    # Configure the LoRA parameters
    config = LoraConfig(
        r=32, # Rank, no. of parameters trained (E.g., for a 512x512 (262144) matrix, if rank = 64, the LoRA adapter uses 512x64 and 64x512 parameters.)
        lora_alpha=64, # Alpha, how much the model adapts to the new training data.
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        use_dora=True
    )

    # Enable gradient checkpointing to reduce memory usage during fine-tuning
    base_model.gradient_checkpointing_enable()

    # Prepare the base model for QLoRA
    base_model = prepare_model_for_kbit_training(base_model)

    # Get the LoRA trainable version of the model (LoRA adapter)
    peft_model = get_peft_model(base_model, config)

    # Define the training arguments
    peft_training_args = TrainingArguments(
        output_dir = output_dir,
        warmup_steps=50, # For the first n steps, learning rate slowly increases
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2, # evaluation batch size
        gradient_accumulation_steps=2, # Updates model every n batch
        num_train_epochs=2,
        learning_rate=5e-4, #(0.00002)
        lr_scheduler_type='cosine',
        optim="paged_adamw_8bit", # Optimizer type used to update weights
        logging_steps=10, # Log the loss output every n steps
        logging_dir="./logs",
        save_strategy="epoch",
        # save_steps=50, # Save model every i steps
        eval_strategy="steps", # evaluation strategy (High GPU RAM)
        eval_steps=10, # evaluation steps (High GPU RAM)
        do_eval=True,
        gradient_checkpointing=True,
        report_to="wandb",
        overwrite_output_dir = 'True',
        group_by_length=True,
        fp16=True)

    # Disable caching to save memory
    peft_model.config.use_cache = False

    # Create the 'Trainer' instance
    peft_trainer = Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=peft_training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    return peft_trainer

def upload_model_to_gcp(local_folder_path, bucket_name, destination_folder_base, time):
    # Generate a timestamped folder name
    destination_folder_name = os.path.join(destination_folder_base, time)

    # Initialize the GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Loop through all files in the folder
    for root, _, files in os.walk(local_folder_path):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)

            # Get the destination path in GCP
            relative_path = os.path.relpath(local_file_path, local_folder_path)
            destination_blob_name = os.path.join(destination_folder_name, relative_path).replace("\\", "/")

            # Upload the file
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{bucket_name}/{destination_blob_name}")

def merge_model(base_model, model_save_path):
    # Get the LoRA adapter
    ft_model = PeftModel.from_pretrained(base_model, model_save_path, torch_dtype=torch.float16, is_trainable=False)

    # Free memory for merging weights
    torch.cuda.empty_cache()

    # Merge the LoRA adapter with the base model and save the merged model
    lora_merged_model = ft_model.merge_and_unload()

    return lora_merged_model

def evaluate_rouge_scores(model, tokenizer, val_dataset):
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Lists to store ROUGE scores
    rouge1_scores = [] # Compares the important words to ground truth
    rouge2_scores = [] # Compares short phrases to ground truth
    rougeL_scores = [] # Compares overall response to ground truth

    # Iterate through the dataset
    for example in val_dataset:
        # Extract context and true response
        context = example["Context"]
        true_response = example["Response"]

        # System prompt 
        messages = [
            {"role": "system", "content": "You are a mental health chatbot"},
            {"role": "user", "content": context}
        ]

        # Tokenize text
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # Keep text as string
            add_generation_prompt=True  # Add generation instructions if needed
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate response 
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.5,
            top_p=0.5
        )

        # Get only the generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the generated response
        generated_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Compute ROUGE scores
        rouge_score = scorer.score(true_response, generated_response)

        # Append scores
        rouge1_scores.append(rouge_score['rouge1'].fmeasure)
        rouge2_scores.append(rouge_score['rouge2'].fmeasure)
        rougeL_scores.append(rouge_score['rougeL'].fmeasure)

    # Calculate average ROUGE scores
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    return {
        "Average ROUGE-1 F-Measure": avg_rouge1,
        "Average ROUGE-2 F-Measure": avg_rouge2,
        "Average ROUGE-L F-Measure": avg_rougeL
    }

def push_model(lora_merged_model, tokenizer, pushed_model_name):
    # Save model and tokenizer
    lora_merged_model.save_pretrained("merged",safe_serialization=True)
    tokenizer.save_pretrained("merged")

    # Push merged model to the hub
    lora_merged_model.push_to_hub(pushed_model_name) # the name of the model you want
    tokenizer.push_to_hub(pushed_model_name)

    print("Model pushed to HuggingFace!")