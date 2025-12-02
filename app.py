import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor
import gym
from gym import spaces
import random
import warnings
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="OptimaFlux", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📦"
)

# ------------------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 5rem;}
    
    /* KPI Card Container */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    padding: 15px;
    height: 110px; /* Forced height for uniformity */
    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* KPI Label (e.g., "Net Profit") */
div[data-testid="stMetric"] label { 
    color: #6E7781; 
    font-size: 0.85rem; 
}

/* KPI Value (e.g., "$50,000") */
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { 
    color: #1F2937; 
    font-size: 1.5rem; 
    font-weight: 700; 
}
    
    /* Primary Action Button */
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 6px rgba(255, 75, 75, 0.2);
    }
    div.stButton > button:first-child:hover {
        background-color: #FF2B2B;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------
# SIDEBAR: CONFIGURATION & DOWNLOAD
# ------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # --- 1. DATA TEMPLATE DOWNLOAD (Moved to Sidebar) ---
    st.subheader("1. Data Source")
    
    def generate_template():
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=90, freq='D'),
            'sales': np.random.randint(20, 100, size=90),
            'price': [25.0] * 90,
            'promo': np.random.choice([0, 1], size=90, p=[0.8, 0.2])
        })
        return df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "📥 Get CSV Template",
        data=generate_template(),
        file_name="retail_data.csv",
        mime="text/csv",
        use_container_width=True,
        help="Download a sample CSV to see the required format."
    )
    
    st.markdown("---")

    # --- 2. ADVANCED SETTINGS ---
    st.subheader("2. Parameters")
    
    with st.expander("Operational Costs", expanded=True):
        LEAD_TIME = st.number_input("Lead Time (Days)", value=5, min_value=1)
        HOLDING_COST = st.number_input("Daily Holding Cost ($)", value=0.5, step=0.1)
        UNIT_COST = st.number_input("Unit Cost ($)", value=15.0, step=1.0)
        STARTING_STOCK = st.number_input("Current Inventory", value=100, min_value=0)

    with st.expander("AI Model Settings"):
        FORECAST_DAYS = st.slider("Forecast Horizon", 14, 90, 30)
        RL_EPISODES = st.slider("AI Training Steps", 100, 2000, 500)

# ------------------------------------------------------------------------
# MAIN HERO SECTION
# ------------------------------------------------------------------------
st.title("📦 OptimaFlux")
st.markdown("#### AI-Driven Demand, Inventory & Pricing Optimization")

st.markdown("---")

# How it Works - Icons
c1, c2, c3 = st.columns(3)
with c1:
    st.image("https://img.icons8.com/color/96/bullish.png", width=60)
    st.markdown("**1. Predict Demand**\n\nProphet analyzes seasonality to forecast future sales.")
with c2:
    st.image("https://img.icons8.com/fluency/96/warehouse.png", width=60)
    st.markdown("**2. Optimize Stock**\n\nGBM predicts optimal inventory levels to prevent stockouts.")
with c3:
    st.image("https://img.icons8.com/color/96/price-tag.png", width=60)
    st.markdown("**3. Dynamic Pricing**\n\nRL Agent adapts prices to maximize profit margins.")

st.markdown("---")

# ------------------------------------------------------------------------
# DATA IMPORT (MAIN STAGE)
# ------------------------------------------------------------------------
st.subheader("📂 Upload Data")
uploaded_file = st.file_uploader("Upload your sales data (CSV)", type=['csv'])

# ------------------------------------------------------------------------
# LOGIC: DATA LOADING
# ------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    if file is None: return None
    try:
        df = pd.read_csv(file)
        df.columns = [c.lower().strip() for c in df.columns]
        
        if not {'date', 'sales', 'price'}.issubset(df.columns):
            st.error("Missing required columns: date, sales, price")
            return None
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        if 'promo' not in df.columns: df['promo'] = 0
        return df
    except Exception as e:
        st.error(f"File Error: {e}")
        return None

df = load_data(uploaded_file)

# ------------------------------------------------------------------------
# LOGIC: AI MODELS
# ------------------------------------------------------------------------

# MODEL 1: PROPHET
@st.cache_data
def run_model_1_demand(data, horizon):
    df_p = data[['date', 'sales', 'promo']].rename(columns={'date': 'ds', 'sales': 'y'})
    m = Prophet(daily_seasonality=True)
    m.add_regressor('promo')
    m.fit(df_p)
    future = m.make_future_dataframe(periods=horizon)
    future['promo'] = 0 
    forecast = m.predict(future)
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon)
    result = result.rename(columns={'ds': 'date', 'yhat': 'forecast_demand'})
    result['forecast_demand'] = result['forecast_demand'].apply(lambda x: max(0, x))
    return result

# MODEL 2: GBM INVENTORY
@st.cache_data
def run_model_2_inventory(history, forecast, lead_time, holding_cost):
    hist_data = history.copy()
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=lead_time)
    hist_data['target_inventory'] = hist_data['sales'].rolling(window=indexer).sum()
    hist_data['lead_time'] = lead_time
    hist_data['holding_cost'] = holding_cost
    
    train_df = hist_data.dropna()
    
    if len(train_df) < 10:
        forecast['recommended_stock'] = forecast['forecast_demand'] * lead_time * 1.5
        return forecast
        
    X = train_df[['lead_time', 'holding_cost', 'promo']]
    y = train_df['target_inventory']
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    future_X = pd.DataFrame({
        'lead_time': [lead_time] * len(forecast),
        'holding_cost': [holding_cost] * len(forecast),
        'promo': [0] * len(forecast)
    })
    
    forecast['recommended_stock'] = model.predict(future_X)
    forecast['recommended_stock'] = forecast['recommended_stock'] * 1.2 # Safety Buffer
    forecast['recommended_stock'] = forecast['recommended_stock'].apply(lambda x: max(0, x))
    
    return forecast

# MODEL 3: RL PRICING
class PricingEnv(gym.Env):
    def __init__(self, data, base_price, unit_cost, start_stock):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.base_price = base_price
        self.unit_cost = unit_cost
        self.start_stock = start_stock
        self.action_space = spaces.Discrete(3) # 0:Lower, 1:Keep, 2:Raise
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)
        self.reset()
        
    def reset(self):
        self.t = 0
        self.stock = self.start_stock
        return np.array([self.stock, self.t])
        
    def step(self, action):
        multipliers = [0.95, 1.0, 1.05] 
        price = self.base_price * multipliers[action]
        
        # Get demand forecast safely
        if self.t < len(self.data):
            base_demand = self.data['forecast_demand'].iloc[self.t]
        else:
            base_demand = 0 
            
        elasticity = -2.5
        actual_demand = base_demand * (1 + elasticity * ((price - self.base_price)/self.base_price))
        actual_demand = max(0, actual_demand)
        
        sales = min(self.stock, actual_demand)
        lost_sales = actual_demand - sales
        
        revenue = sales * price
        cogs = sales * self.unit_cost
        profit = revenue - cogs
        
        self.stock -= sales
        
        if self.t < len(self.data):
            replenish = self.data['recommended_stock'].iloc[self.t]
            self.stock += replenish
            
        self.t += 1
        done = self.t >= len(self.data)
        
        return np.array([self.stock, self.t]), profit, done, {
            'price': price, 'sales': sales, 'revenue': revenue, 
            'stock': self.stock, 'demand': actual_demand, 'lost_sales': lost_sales,
            'profit': profit
        }

def run_model_3_pricing(plan_df, base_price, unit_cost, start_stock, episodes):
    if plan_df.empty: return pd.DataFrame()

    env = PricingEnv(plan_df, base_price, unit_cost, start_stock)
    q_table = {}
    
    # Train
    for _ in range(episodes):
        state = env.reset()
        state_key = int(state[0] / 10) 
        done = False
        while not done:
            if state_key not in q_table: q_table[state_key] = np.zeros(3)
            if random.random() < 0.15: action = env.action_space.sample()
            else: action = np.argmax(q_table[state_key])
            next_state, reward, done, _ = env.step(action)
            next_key = int(next_state[0] / 10)
            if next_key not in q_table: q_table[next_key] = np.zeros(3)
            q_table[state_key][action] += 0.1 * (reward + 0.9 * np.max(q_table[next_key]) - q_table[state_key][action])
            state_key = next_key
            
    # Inference
    obs = env.reset()
    history = []
    for _ in range(len(plan_df)):
        key = int(obs[0] / 10)
        action = np.argmax(q_table.get(key, [0, 1, 0])) 
        obs, _, _, info = env.step(action)
        history.append(info)
        
    return pd.DataFrame(history)

# ------------------------------------------------------------------------
# 4. EXECUTION
# ------------------------------------------------------------------------

if df is not None:
    st.markdown("###")
    
    # Central Action Button
    if st.button("🚀 Launch AI Pipeline", type="primary", use_container_width=True):
        
        with st.status("🤖 Running AI Models...", expanded=True) as status:
            st.write("📈 Phase 1: Forecasting Demand (Prophet)...")
            forecast_df = run_model_1_demand(df, FORECAST_DAYS)
            time.sleep(0.3)
            
            st.write("📦 Phase 2: Optimizing Inventory (GBM)...")
            inventory_df = run_model_2_inventory(df, forecast_df, LEAD_TIME, HOLDING_COST)
            time.sleep(0.3)
            
            st.write("💰 Phase 3: Optimizing Pricing (Reinforcement Learning)...")
            avg_price = df['price'].mean()
            results = run_model_3_pricing(inventory_df, avg_price, UNIT_COST, STARTING_STOCK, RL_EPISODES)
            results['date'] = inventory_df['date'].values
            
            status.update(label="✅ Optimization Complete!", state="complete", expanded=False)

        st.markdown("---")
        
        # KPI ROW
        st.subheader("🏆 Performance Overview")
        
        total_profit = results['profit'].sum()
        total_rev = results['revenue'].sum()
        avg_margin = (total_profit / total_rev) * 100 if total_rev > 0 else 0
        stockouts = len(results[results['lost_sales'] > 0])
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Projected Revenue", f"${total_rev:,.0f}", delta="Forecast")
        k2.metric("Net Profit", f"${total_profit:,.0f}", delta=f"{avg_margin:.1f}% Margin")
        k3.metric("Stockout Risk", f"{stockouts} Days", delta_color="inverse")
        k4.metric("Ending Stock", f"{int(results['stock'].iloc[-1])} Units")

        st.markdown("###") 

        # CHART TABS
        tab1, tab2, tab3 = st.tabs(["📈 Demand & Inventory", "🏷️ Dynamic Pricing", "💾 Raw Data"])

        with tab1:
            st.markdown("##### Forecast vs. Optimized Stock")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(go.Scatter(
                x=results['date'], y=results['stock'], name="Inventory Level",
                fill='tozeroy', line=dict(color='#636EFA', width=1)
            ), secondary_y=True)
            
            fig.add_trace(go.Scatter(
                x=results['date'], y=inventory_df['forecast_demand'], name="Forecasted Demand",
                line=dict(color='#00CC96', width=3)
            ), secondary_y=False)

            fig.update_layout(height=450, hovermode="x unified", template="plotly_white")
            fig.update_yaxes(title_text="Demand (Units)", secondary_y=False)
            fig.update_yaxes(title_text="Stock (Units)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("##### Dynamic Price Adaptation")
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig2.add_trace(go.Bar(
                x=results['date'], y=results['revenue'], name="Revenue",
                marker_color='#AB63FA', opacity=0.6
            ), secondary_y=False)
            
            fig2.add_trace(go.Scatter(
                x=results['date'], y=results['price'], name="Optimal Price",
                mode='lines+markers', line=dict(color='#EF553B', width=2)
            ), secondary_y=True)

            fig2.update_layout(height=450, hovermode="x unified", template="plotly_white")
            fig2.update_yaxes(title_text="Daily Revenue ($)", secondary_y=False)
            fig2.update_yaxes(title_text="Unit Price ($)", secondary_y=True)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.dataframe(results, use_container_width=True)
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv, "ai_results.csv", "text/csv")

else:
    st.info("👆 Download the template from the sidebar or upload your own data to begin.")