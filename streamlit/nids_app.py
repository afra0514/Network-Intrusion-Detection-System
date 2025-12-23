import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import time 
st.set_page_config(
    page_title="SecureNet AI",
    page_icon="🛡️",
    layout="centered"
)
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    /* HEADER BOX */
    .header-box {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    /* INPUT FORM STYLING */
    .stNumberInput > label {
        font-weight: 600;
        color: #333;
    }
    /* BUTTON STYLING */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #0052D4 0%, #4364F7 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 12px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 82, 212, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.01);
        box-shadow: 0 6px 12px rgba(0, 82, 212, 0.3);
    }
    /* RESULT CARDS - Centered Prediction */
    .result-safe {
        background-color: #d1fae5; /* Soft Green */
        color: #065f46;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #10b981;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        animation: fadeIn 0.5s;
    }
    .result-danger {
        background-color: #fee2e2; /* Soft Red */
        color: #991b1b;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #ef4444;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        animation: fadeIn 0.5s;
    }
    /* PREDICTION LABEL STYLE */
    .pred-label {
        font-size: 22px; 
        font-weight: bold; 
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* FOOTER */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #666;
        text-align: center;
        padding: 12px;
        border-top: 1px solid #eaeaea;
        font-size: 13px;
        z-index: 999;
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
st.markdown("""
    <div class="header-box">
        <h1 style="margin:0; font-size: 2.2rem;">🛡️ SecureNet AI</h1>
        <p style="margin-top:8px; font-size: 1rem; opacity: 0.9;">
            Intelligent Network Intrusion Detection System
        </p>
    </div>
""", unsafe_allow_html=True) 
@st.cache_resource
def load_assets():
    try:
        model = load_model('hybrid_model.keras')
        scaler = joblib.load('minmax_scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, scaler, le
    except:
        return None, None, None
model, scaler, le = load_assets()
if not model:
    st.error("🚨 Critical Error: Model files missing! Please ensure 'hybrid_model.keras', 'minmax_scaler.pkl', and 'label_encoder.pkl' are in the directory.")
    st.stop() 
st.markdown("### 📡 Traffic Configuration") 
all_features = [
    'Bwd Packet Length Std', 'Total Length of Bwd Packets', 'Flow IAT Std',
    'Fwd IAT Std', 'Total Length of Fwd Packets', 'Destination Port',
    'Init_Win_bytes_backward', 'Total Fwd Packets', 'Bwd Packets/s',
    'Fwd Packet Length Std', 'Flow Packets/s', 'PSH Flag Count',
    'Init_Win_bytes_forward', 'Flow Duration', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Total Backward Packets', 'Fwd IAT Min',
    'ACK Flag Count', 'Flow Bytes/s'
] 
with st.form("prediction_form"):
    st.write("Input the network parameters below:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🌐 Network Details**")
        p1 = st.number_input("Destination Port", 0, 65535, 80)
        p2 = st.number_input("Flow Duration (ms)", 0.0, value=1000.0)
        p3 = st.number_input("Flow Bytes/s", 0.0, value=500.0)
    with col2:
        st.markdown("**📦 Packet Stats**")
        p4 = st.number_input("Total Fwd Packets", 0, value=10)
        p5 = st.number_input("Total Length Fwd (Bytes)", 0.0, value=500.0)
        p6 = st.number_input("Bwd Packet Length Std", 0.0, value=0.0)
    st.markdown("---")
    submitted = st.form_submit_button("🚀 SCAN TRAFFIC NOW") 
if submitted:
    with st.spinner("🧠 Analyzing packet signatures..."):
        time.sleep(0.8) 
        try:
            input_values = {
                'Destination Port': p1, 'Flow Duration': p2, 'Flow Bytes/s': p3,
                'Total Fwd Packets': p4, 'Total Length of Fwd Packets': p5, 
                'Bwd Packet Length Std': p6
            } 
            final_input = {feature: 0.0 for feature in all_features}
            for key, value in input_values.items():
                final_input[key] = value
            df_input = pd.DataFrame([final_input]) 
            scaled_input = scaler.transform(df_input)
            reshaped_input = scaled_input.reshape(1, 20, 1)
            pred_probs = model.predict(reshaped_input)
            prediction_index = np.argmax(pred_probs, axis=1)[0]
            result_label = le.inverse_transform([prediction_index])[0]
            st.write("") 
            if result_label.upper() in ['BENIGN', 'NORMAL']:
                st.markdown(f"""
                <div class="result-safe">
                    <h2 style='margin:0'>SYSTEM SECURE</h2>
                    <p class="pred-label">Normal Behavior</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-danger">
                    <h2 style='margin:0'>THREAT DETECTED</h2>
                    <p class="pred-label">{result_label}</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")
st.markdown("""
    <div class="footer">
        <p>🔒 <b>SecureNet AI</b> |All Rights REserved| © 2025</p>
    </div>
""", unsafe_allow_html=True)