# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /train_model

# Copy the current directory (including FastAPI app and model) into the container
COPY . /train_model

# Install the dependencies required for FastAPI and the model
RUN pip install --no-cache-dir fastapi uvicorn joblib numpy scikit-learn joblib pandas imblearn

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
