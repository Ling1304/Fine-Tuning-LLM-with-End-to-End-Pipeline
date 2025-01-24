# Use the official Apache Airflow image as the base image
FROM apache/airflow:2.10.4

# Set environment variables for Airflow
ENV AIRFLOW_HOME=/opt/airflow

# Copy requirements.txt
COPY requirements.txt /requirements.txt

# Switch to the airflow user
USER airflow

# Install Python and requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /requirements.txt
