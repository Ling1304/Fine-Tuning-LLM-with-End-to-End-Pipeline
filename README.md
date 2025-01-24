# Fine-Tuning a Mental Health LLM with End-to-End Pipeline using Apache Airflow

This repository focuses on fine-tuning a **Large Language Model (LLM)** to assist individuals with their mental health and provide support for live call centers. Using **PEFT (Parameter-Efficient Fine-Tuning)** techniques such as **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized Low-Rank Adaptation)**, the model achieves high performance with minimal resource requirements.

The implementation utilizes the **Amod/mental_health_counseling_conversations** dataset from Hugging Face and integrates **BitsAndBytes** for 4-bit double quantization, ensuring scalability even on hardware with limited computational resources.

---

## 🛠️ Fine Tuning Process  

The fine-tuning process involves a series of carefully orchestrated steps:  

### 1. **Dataset**  
- The `Amod/mental_health_counseling_conversations` dataset is used for training, containing real-world examples of counseling conversations.  
- The dataset is preprocessed and formatted to align with the model’s input structure.  

### 2. **Model Quantization**  
- The model is loaded using **BitsAndBytes** in **4-bit double quantized mode**, significantly reducing memory and computational overhead while maintaining performance.  

### 3. **PEFT Techniques**  
- **LoRA**: Efficiently tunes a subset of model parameters using low-rank matrices, reducing the number of parameters requiring updates.  
- **QLoRA**: Further optimizes memory usage by combining quantization with LoRA, enabling high performance on smaller hardware setups.  

### 4. **Training Process**  
- The model is fine-tuned using carefully selected hyperparameters (e.g., learning rate, batch size).  
- Loss minimization and evaluation metrics guide the optimization process.  

### 5. **Finalization and Deployment**  
- After fine-tuning, the **LoRA adapter** is merged with the base model to create the final version.  
- The fine-tuned model is pushed to Hugging Face for public access and deployment.  

---

## 📂 Repository Structure  

### `src` Folder  
- **`data_preprocessing.ipynb`**  
  Handles preprocessing tasks for the dataset, including tokenization and formatting for training. (Note that only 20% of the data is used for training due to memory constraints)

- **`train.ipynb`**  
  Script for fine-tuning the model. Includes training loop, optimizer configuration, and checkpoint saving.   

### `models` Folder  
- Contains the trained model weights after fine-tuning.  

---

## Key Features  

- **Mental Health Focus**: Specifically fine-tuned for counseling and support conversations.  
- **Resource Efficiency**: LoRA and QLoRA enable fine-tuning with reduced computational requirements.  
- **Scalability**: Can be deployed on hardware with limited memory and processing power.  
- **Reproducibility**: Includes a detailed step-by-step workflow for easy replication.  

---

Stay tuned for updates and enhancements! Contributions and feedback are welcome to improve this project further.  
