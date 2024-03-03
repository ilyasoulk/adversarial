# Use an official PyTorch image as the base
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the script file into the container
COPY phi.py .

# Install Python dependencies
RUN pip install --no-cache-dir transformers datasets

# Run the script when the container launches
CMD ["python", "phi.py"]
