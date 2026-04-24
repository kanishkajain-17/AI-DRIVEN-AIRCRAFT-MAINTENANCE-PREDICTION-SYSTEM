# AI-DRIVEN-AIRCRAFT-MAINTENANCE-PREDICTION-SYSTEM
This repository implements an AI-driven maintenance system that focuses on Prognostics and Health Management (PHM). Rather than relying on calendar-based schedules, this system uses Deep Learning and Anomaly Detection to calculate the exact Remaining Useful Life (RUL) of turbofan engines and clusters them into actionable health categories.

# Aircraft Maintenance Health Status Prediction using LSTM, Isolation Forest & Clustering

## 📌 Project Overview
This project is developed for **Aircraft Predictive Maintenance, Anomaly Detection, and Health Status Prediction** using Machine Learning and Deep Learning techniques.  

The system analyzes aircraft maintenance and operational data to:

- Predict future failures using **LSTM**
- Detect abnormal behavior using **Isolation Forest**
- Classify aircraft condition using **Clustering**

The dataset used in this project is the **CMAPSS Dataset**, collected from Kaggle and preprocessed for model training and evaluation.
It helps improve aircraft safety, reduce downtime, and support smart maintenance decisions.

---

## 📂 Dataset Information

### 📊 Dataset Used: CMAPSS Dataset

The project uses the **CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset, available on Kaggle.

It contains simulated aircraft engine degradation data with multiple sensor readings and operational settings.

### Dataset Includes:

- Engine unit number
- Time cycles
- Operational settings
- Multiple sensor measurements
- Engine degradation behavior
- Remaining Useful Life (RUL) prediction data

### Data Source:

Kaggle – CMAPSS Turbofan Engine Degradation Dataset

---

## 🛠️ Data Preprocessing Performed

Before training models, the dataset was processed using the following steps:

- Removed unnecessary columns
- Handled missing values
- Feature selection
- Sensor normalization
- Scaling using MinMaxScaler / StandardScaler
- Sequence generation for LSTM


---

## 🎯 Key Features

- Data preprocessing and normalization  
- Anomaly detection using Isolation Forest  
- Sequential trend prediction using LSTM  
- **Health status prediction using clustering**
- Categorization into:
  - ✅ Safe  
  - ⚠️ Warning  
  - 🚨 Critical  
- Visualization dashboard  
- Backend & frontend integration

---

## 📂 Project Structure

### 📁 Folders

- `data/` → Processed datasets  
- `models/` → Saved ML/DL models  
- `outputs/` → Predictions, health reports  


### 📄 Python Files

- `backend.py` → Backend server/API  
- `frontend.py` → Frontend interface  
- `train_model.py` → Model training  (Isolation Forest)
- `test_backend.py` → Testing scripts  

### 📘 Jupyter Notebooks

- `anamoly_preprocessed_train.ipynb` → Training notebook  
- `LSTM_dataset_preprocessed(anamoly_phase).ipynb` → Dataset preprocessing  
- `LSTM_testing_preprocessed(code).ipynb` → Model testing  

### 📑 Reports

- `CSD009 Report.pdf` → Final Project Report  
- `SRS_SDS_Aircraft_Maintenance_Project.pdf` → SRS/SDS Documentation  

---

## 🧠 Models & Algorithms Used

## 1️⃣ LSTM (Long Short-Term Memory)

Used to learn historical sequential patterns and predict future system behavior.

## 2️⃣ Isolation Forest

Used to identify unusual or abnormal aircraft operational records.

### Output:

- `1` → Normal  
- `-1` → Anomaly

## 3️⃣ Clustering Algorithm

Used to group aircraft health conditions into categories.

Possible algorithms:

- K-Means Clustering  


### Predicted Health Status:

| Cluster | Status |
|--------|--------|
| Cluster 1 | ✅ Safe |
| Cluster 2 | ⚠️ Warning |
| Cluster 3 | 🚨 Critical |

---

## ⚙️ Technologies Used

- Python  
- Scikit-learn  
- TensorFlow / Keras  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Streamlit  

---

## ▶️ How to Run Project

### 1️⃣ Activate Virtual Environment

```bash
venv\Scripts\activate
```

### 2️⃣ Train Model

```bash
python train_model.py
```

### 3️⃣ Run LSTM

```bash
python LSTM_dataset_preprocessed.py
```

### 4️⃣ Run Frontend

```bash
streamlit run frontend.py
```

---

## 📈 Output

- Failure prediction  
- Anomaly detection results  
- Health status classification  
- Graphical analysis  


---

## 🚀 Future Scope

- Real-time aircraft monitoring  
- IoT sensor data integration  
- Live warning alerts  
- Cloud deployment  
- Advanced AI maintenance system  

---

## 👩‍💻 Members
Anshika Nagpal
Charu Solanki
Ishita Jain
Kanishka Jain
Kaushiki Mishra


B.Tech 3rs Year Project

```
