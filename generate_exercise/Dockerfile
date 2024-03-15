# Use an official PyTorch image as the base
FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y git

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the script file into the container
COPY generate.py .

# Install Python dependencies
RUN pip install datasets peft
RUN pip install git+https://github.com/huggingface/transformers
# Add your own token here
ENV TOKEN=write_your_own_token_here

# Run the script when the container launches
CMD ["python", "generate.py"]
