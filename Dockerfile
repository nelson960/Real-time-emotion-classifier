# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy everything into the container
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["OPENCV_AVFOUNDATION_SKIP_AUTH=1", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
