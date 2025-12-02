# OptimaFlux
# 📦 Supply Chain Control Tower

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A fully vertical **AI-driven Supply Chain Optimization** application. This dashboard integrates Demand Forecasting, Inventory Management, and Dynamic Pricing into a single, cohesive pipeline to maximize retail profitability.

## 🚀 Key Features

The application runs a **3-Stage Vertical Pipeline**:

1.  **📈 Demand Forecasting (Time Series)**
    * **Model:** Meta's [Prophet](https://facebook.github.io/prophet/).
    * **Function:** Detects seasonality, trends, and promotion effects to predict future daily demand.

2.  **📦 Inventory Optimization (Supervised Learning)**
    * **Model:** Gradient Boosting Regressor (GBM).
    * **Function:** Calculates optimal reorder quantities based on forecasted demand, lead times, and holding costs to prevent stockouts and overstocking.

3.  **💰 Dynamic Pricing (Reinforcement Learning)**
    * **Model:** Q-Learning Agent (Tabular RL).
    * **Function:** Simulates market conditions and learns the optimal price point ($) to maximize revenue while respecting inventory constraints.

---

## 🛠️ Installation & Local Setup

Follow these steps to run the app on your own machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/supply-chain-optimizer.git](https://github.com/your-username/supply-chain-optimizer.git)
cd supply-chain-optimizer
```
### 2. Enable virtual environment
``` bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```
### 3. Install dependencies and Run Streamlit
``` bash
pip install -r requirements.txt
streamlit run app.py
```
