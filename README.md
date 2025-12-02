# End-to-End Telco Churn Prediction System 🚀

**Status:** Production-Ready 🟢 | **Role:** MLOps Engineer

## 📖 Overview
A full-stack machine learning system that predicts customer churn in real-time. Unlike standard tutorials, this project features a **custom-built drift detection engine** (using KS & Chi-Square tests), a hardened **Docker/Kubernetes** deployment, and automated **CI/CD pipelines**.

## 🏗️ Architecture
* **Data Lakehouse:** Databricks (PySpark)
* **Model:** Random Forest (Class-Weighted)
* **Monitoring:** Custom Statistical Engine (KS Test / Chi-Square)
* **Containerization:** Docker (Python 3.9 Slim)
* **Orchestration:** Kubernetes (Minikube + Docker Driver)
* **CI/CD:** GitHub Actions -> Docker Hub
* **Dashboard:** Power BI

## 🔧 Key Technical Implementation

### 1. Advanced Model Training
* Implemented **Dual-Threshold Logic**:
    * *Optimal Threshold (0.47):* Maximized Youden’s J statistic.
    * *Business Threshold (0.35):* Prioritized Recall to capture 15% more at-risk customers.
* Logged 10+ custom metrics to **MLflow**.

### 2. Custom Drift Detection (No External Libs)
Instead of using off-the-shelf tools, I engineered a statistical monitoring script:
* **Numerical Features:** monitored via **Kolmogorov-Smirnov (KS) Test**.
* **Categorical Features:** monitored via **Chi-Square Test**.
* Alerts triggered if >30% of features show $p < 0.05$.

### 3. Production Deployment & DevOps
* **API:** Developed a Flask microservice handling JSON payloads.
* **Dependency Management:** Solved critical `scikit-learn 1.5` vs `1.3` serialization conflicts by pinning environment versions.
* **Kubernetes:** Deployed High-Availability replicas (2x) using a `NodePort` service for external access.
* **CI/CD:** Automated the build-test-push cycle. Any code push to `main` triggers GitHub Actions to update the Docker Hub registry.

## 📊 Business Dashboard (Power BI)
Connected directly to Databricks Delta Tables to track:
* Real-time Model Drift Score.
* Churn Rate by Contract Type.
* Alert Thresholds (Red line at 0.3 drift score).

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone [https://github.com/YOUR_USERNAME/churn-api.git](https://github.com/YOUR_USERNAME/churn-api.git)

# 2. Build the Docker Container
docker build -t churn-api:latest .

# 3. Run the Microservice
docker run -p 5000:5000 churn-api:latest

# 4. Test Endpoint
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '[{"gender":1, "SeniorCitizen":0, ...}]'