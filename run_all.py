import os

print("\n🚀 Generating dataset...")
os.system("python data.py")

print("\n🧠 Training model...")
os.system("python train_model.py")

print("\n⚡ Launching Streamlit app...")
os.system("streamlit run app.py")