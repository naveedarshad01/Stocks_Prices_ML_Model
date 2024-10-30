# Use a lightweight Python image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install Python dependencies
COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

# Copy the source code into the container
COPY src/ .

# Run the application
CMD ["python", "app.py"]
