# 🎓 Machine Learning Capstone Project App

An end-to-end Streamlit machine learning application for the internship capstone. The project keeps the original dashboard design while replacing the demo calculations with a genuine serialized scikit-learn inference pipeline and dataset-driven analytics.

## 🚀 Project Features
- Multi-page Streamlit dashboard
- Exploratory data analysis from a real CSV file included in the repository
- 1,500 customer-account records
- Logistic Regression model with StandardScaler
- Serialized trained model (`models/model.pkl`)
- Cached model loading with `st.cache_resource`
- Cached dataset loading with `st.cache_data`
- Live probability prediction from user inputs
- Accuracy, precision, recall, F1 and ROC-AUC metrics
- GitHub-ready project structure

## 📁 Project Structure
```text
capstone-ml-dashboard-main/
├── data/
│   └── customer_accounts.csv
├── models/
│   ├── model.pkl
│   └── metrics.json
├── pages/
│   ├── data_analysis.py
│   └── model_prediction.py
├── Home.py
├── train_model.py
├── requirements.txt
└── readme.md
```

## 🧠 Machine Learning Pipeline
The application predicts customer-account default risk using these features:
- Age
- Annual income
- Credit score
- Loan amount
- Employment years
- Debt-to-income ratio

The model is a scikit-learn Pipeline containing:
1. `StandardScaler` for feature scaling
2. `LogisticRegression` for binary classification

The trained pipeline is serialized to `models/model.pkl` and loaded by the Streamlit prediction page.

## 📊 Dataset
The included dataset contains 1,500 synthetic customer-account records. It was generated reproducibly with random seed 42 specifically for this internship capstone, so no external dataset download is required for deployment.

## ⚙️ Local Setup
1. Clone this repository.
2. Create and activate a virtual environment (recommended).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the Streamlit application:

```bash
streamlit run Home.py
```

5. Open the local URL displayed by Streamlit.

## 🌐 Deployment
The project is suitable for deployment on Streamlit Community Cloud. Connect the public GitHub repository, select `Home.py` as the main file, and deploy.

## 📈 Model Evaluation
The latest evaluation metrics are stored in `models/metrics.json` and displayed automatically in the Data Analysis page.

## 🎯 Capstone Deliverables
- Public GitHub repository: required
- Live Streamlit/Demo URL: required
- Serialized trained model: included
- Interactive Streamlit interface: included
- Dataset-driven analytics: included
- Setup documentation: included

## 👩‍💻 Internship Capstone
This project integrates model training, serialization, caching, interactive Streamlit widgets, data analysis and live deployment into one end-to-end application.
