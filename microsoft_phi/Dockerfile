# Use an official PyTorch image as the base
FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y git

# Set the working directory in the container
WORKDIR /usr/src/app

RUN mkdir -p /usr/src/app/generated_outputs

# Copy the script file into the container
# COPY phi.py .
COPY lora_params.py .
# Install Python dependencies
RUN pip install datasets peft
RUN pip install git+https://github.com/huggingface/transformers.git

# Add your own token here
ENV TOKEN=

# Run the script when the container launches
# CMD ["python", "phi.py"]
CMD ["python", "lora_params.py"]
