# FROM gcr.io/deeplearning-platform-release/pytorch-cpu.1-4
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-4

WORKDIR /root

RUN pip install transformers==2.11.0 nlp==0.2.0 google-cloud-storage

COPY trainer/task.py ./trainer/task.py

# Sets up the entry point to invoke the trainer
ENTRYPOINT ["python", "trainer/task.py"]