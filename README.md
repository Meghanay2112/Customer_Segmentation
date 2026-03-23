# 🧠 Customer Segmentation — Deep Learning App

A production-grade Streamlit app for customer segmentation using a deep neural network.

## Features
- **Multi-class DNN** with BatchNorm, Dropout, L2 regularization, AdamW optimizer
- **5 Customer Segments**: Champions, Loyal, Potential Loyalists, At Risk, Lost/Churned
- **10 Features**: Age, Income, Spending Score, Recency, Frequency, Monetary, Online Ratio, Satisfaction, Loyalty Years, Products Bought
- **Confusion Matrix** (left panel) with normalized rates
- **PCA Cluster Visualization**
- **Training Curves** (accuracy + loss)
- **Permutation Feature Importance**
- **Live Customer Predictor** — enter values and get instant segment prediction

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** ✅

---

## 🎛️ Configurable Parameters (Sidebar)
| Parameter | Range | Default |
|---|---|---|
| Dataset Size | 1,000 – 10,000 | 5,000 |
| Test Split | 10% – 40% | 20% |
| Network Preset | Compact / Standard / Deep / Ultra | Standard |
| Dropout Rate | 0.10 – 0.50 | 0.25 |
| Max Epochs | 20 – 200 | 80 |

---

## 📊 Expected Accuracy
| Preset | Expected Accuracy |
|---|---|
| Compact | ~92–94% |
| Standard | ~94–96% |
| Deep | ~95–97% |
| Ultra | ~96–98% |

---

## Architecture
```
Input (10) → BatchNorm → [Dense → BN → Swish → Dropout] × N → Softmax (5)
```
- Optimizer: AdamW (weight_decay=1e-4)
- LR Scheduler: ReduceLROnPlateau
- Early Stopping: patience=12 on val_accuracy
