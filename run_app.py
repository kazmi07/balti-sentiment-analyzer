"""
One-click launcher for Balti Sentiment Analysis Web App
Run this script to train model (if needed) and launch the app
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def train_model():
    """Train the model if it doesn't exist"""
    if not os.path.exists('balti_best_model.pkl'):
        print("🤖 Training model...")
        subprocess.check_call([sys.executable, "model_train.py"])
    else:
        print("✅ Model already exists. Skipping training...")

def launch_app():
    """Launch Streamlit app"""
    print("🚀 Launching Balti Sentiment Analyzer...")
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    print("="*50)
    print("BALTI SENTIMENT ANALYSIS APP")
    print("="*50)
    
    # Check if requirements are installed
    try:
        import streamlit
    except ImportError:
        install_requirements()
    
    # Train model if needed
    train_model()
    
    # Launch app
    launch_app()