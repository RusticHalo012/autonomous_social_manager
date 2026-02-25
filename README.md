🚀 Autonomous Social Media Management System

AI-Based Engagement Prediction & Post Scheduling (Offline Version)

📌 Overview

The Autonomous Social Media Management System (ASMS) is a machine learning–driven backend application that:

Compresses historical audience engagement data

Learns latent performance patterns

Predicts engagement scores

Optimizes posting time automatically

Exposes REST endpoints for intelligent scheduling

This version works without real Instagram/Twitter API integration and focuses purely on AI-driven scheduling logic.

🧠 Core Idea

Instead of manually choosing posting times, the system:

Compresses engagement signals using an Autoencoder

Learns performance behavior using XGBoost

Simulates all 24 time slots

Selects the optimal posting hour

Returns predicted engagement score
🏗️ Architecture
Mock Data Generator
        ↓
Autoencoder (Feature Compression)
        ↓
XGBoost Engagement Predictor
        ↓
Scheduling Optimizer (24-Hour Simulation)
        ↓
FastAPI REST Endpoint
📂 Project Structure
autonomous_social_manager/
│
├── main.py
├── data.py
├── compression.py
├── predictor.py
├── scheduler.py
├── feedback.py
└── requirements.txt
⚙️ Technologies Used

Python 3.10+

FastAPI

PyTorch (Autoencoder)

XGBoost

Scikit-learn

NumPy / Pandas

📦 Installation
1️⃣ Clone or Create Project
git clone <your-repo-url>
cd autonomous_social_manager
2️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Running the Project

Run:

python main.py

Server will start at:

http://127.0.0.1:8000

📊 How It Works
1️⃣ Data Simulation

Generates historical engagement data:

Likes

Comments

Shares

Posting hour

Content type

Sentiment

2️⃣ Feature Compression

A PyTorch Autoencoder reduces 6-dimensional features into a 3-dimensional latent vector.

Purpose:

Reduce noise

Capture latent engagement patterns

Improve model efficiency

3️⃣ Engagement Prediction

An XGBoost regressor learns:

Compressed Features → Engagement Score
4️⃣ Scheduling Optimization

For a given post:

System simulates all 24 hours

Predicts engagement for each

Selects hour with highest score

📈 What This Demonstrates

Representation learning

Dimensionality reduction

Supervised regression modeling

Optimization strategy

AI-powered REST backend

Modular ML architecture

🎯 Use Cases

AI/ML portfolio project

Resume-ready backend AI system

Demonstration of model compression

Academic project

Base for enterprise scheduler

🔮 Future Enhancements

Real Instagram/Twitter API integration

Reinforcement Learning scheduler

Real-time feedback learning

User clustering

Trend detection

Database persistence

Docker deployment

Web dashboard
