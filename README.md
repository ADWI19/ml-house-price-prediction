# 🏠 AI Real Estate Price Predictor

A machine learning web application that predicts house prices based on property features like area, number of bedrooms, and bathrooms.  
Built with a complete ML pipeline and an interactive Streamlit dashboard.

---

## 🚀 Live Demo
Link - Streamlit Cloud Link: https://ml-house-price-ai-prediction-577784.streamlit.app

---


---

## 📊 Features

- 🧠 Predict house prices instantly using ML models  
- 📈 Linear Regression vs Random Forest comparison  
- 📊 Interactive visualizations (scatter plot + correlation heatmap)  
- 🎨 Modern dark-themed Streamlit UI  
- ⚙️ Feature scaling for improved prediction accuracy  
- 📁 Fully reproducible ML pipeline  

---

## 🧠 Tech Stack

- Python 🐍  
- Streamlit ⚡  
- Scikit-learn 🤖  
- Pandas 📊  
- NumPy 🔢  
- Matplotlib 📉  
- Seaborn 🌊  

---

## 📁 Project Structure

house-price-project/
│
├── app.py # Streamlit web app
├── data.py # Synthetic dataset generator
├── train_model.py # Model training pipeline
├── run_all.py # (optional) automation script
│
├── data.csv # Generated dataset
├── lr_model.pkl # Linear Regression model
├── rf_model.pkl # Random Forest model
├── scaler.pkl # Feature scaler
│
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── assets/
└── banner.png # Optional UI image

---

## ⚙️ How It Works

Data Generation → Preprocessing → Feature Scaling → Model Training → Prediction UI

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/house-price-ai.git
cd house-price-ai

---

### 2️⃣ Install dependencies
pip install -r requirements.txt

---

### 3️⃣ Generate dataset
python data.py

---

### 4️⃣ Train models
python train_model.py

---

### 5️⃣ Run the app
streamlit run app.py

---

📈 Model Details

Linear Regression → baseline model for trend estimation
Random Forest Regressor → handles non-linearity & improves accuracy
StandardScaler → normalizes features for stable training

---

📊 Visualizations Included
Area vs Price trend analysis
Correlation heatmap of features
Model comparison (LR vs RF predictions)

---

💡 Future Improvements
🌍 Add location-based pricing system
🗺️ Integrate map visualization (Plotly / Folium)
🌐 Deploy backend API using FastAPI
🔐 Add login + user dashboard system
☁️ Deploy full-stack SaaS version

---

👨‍💻 Author

Adwita Chaudhary
Aspiring Data Science / AI ML Engineer

---

⭐ If you like this project
Give it a star ⭐ on GitHub and feel free to fork it for improvements.
