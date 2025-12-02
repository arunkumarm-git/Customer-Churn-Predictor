# 1. Python version
FROM python:3.9-slim

# 2. Create a folder inside the container
WORKDIR /app

# 3. Copy requirements first (to cache dependencies)
COPY requirements_production.txt .

# 4. Install libraries inside the container
# We add scikit-learn==1.3.0 to match your Databricks version
RUN pip install --no-cache-dir -r requirements_production.txt

# 5. Copy the rest of your code (app.py and model.pkl)
COPY . .

# 6. Open port 5000 (The door for traffic)
EXPOSE 5000

# 7. The command to run when the container starts
CMD ["python", "app.py"]