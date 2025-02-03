import streamlit as st 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Try importing plotly and handle missing module gracefully
# Verify if plotly is installed
try:
    import plotly.express as px
    plotly_available = True
    st.success("✅ Plotly is installed successfully!")
except ModuleNotFoundError:
    plotly_available = False
    st.warning("⚠️ Plotly is not installed. Run `pip install plotly` in your local environment.")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset with error handling
@st.cache_data
def load_data():
    path = "Lead Scoring.csv"
    try:
        df = pd.read_csv(path)
        if df.empty:
            st.error("Dataset is empty. Please check the data source.")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

df = load_data()

st.title("🚀 AI-Driven Lead Scoring and ABM Optimization System")

# Ensure dataset is loaded
if df.empty:
    st.warning("No data available. Please check the dataset source.")
    st.stop()

# Data Overview
st.sidebar.header("📊 Data Overview")
if st.sidebar.checkbox("Show Raw Data"):
    st.write(df.head())

# Data Summary
st.sidebar.subheader("📌 Dataset Info")
st.sidebar.write(f"🔹 Rows: {df.shape[0]}")
st.sidebar.write(f"🔹 Columns: {df.shape[1]}")
st.sidebar.write("🔹 Available Columns:", df.columns.tolist())

# Ensure 'Converted' column exists and is numeric
if "Converted" in df.columns:
    # Convert to numeric to avoid Seaborn errors
    df["Converted"] = pd.to_numeric(df["Converted"], errors="coerce")

    if df["Converted"].isna().all():
        st.error("The 'Converted' column contains only NaN values. Check dataset.")
        st.stop()

    st.subheader("📈 Lead Conversion Rate")
    fig, ax = plt.subplots()
    sns.countplot(x='Converted', data=df, ax=ax)
    st.pyplot(fig)
else:
    st.error("The 'Converted' column is missing in the dataset. Cannot generate the plot.")
    st.stop()

# Feature Importance (Using RandomForest for simplicity)
st.subheader("💡 Feature Importance Analysis")

# Handle missing values safely
df.dropna(inplace=True)

# Convert categorical columns to numeric
for col in df.select_dtypes(include=['object']).columns:
    if col != "Converted":  # Exclude target variable if it's categorical
        df[col] = LabelEncoder().fit_transform(df[col])

# Ensure dataset is still valid after preprocessing
if df.empty:
    st.error("Dataset became empty after preprocessing. Please check the data.")
    st.stop()

X = df.drop(columns=["Converted"])  # Features
y = df["Converted"]  # Target

# Split dataset safely
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except ValueError as e:
    st.error(f"Error splitting dataset: {e}")
    st.stop()

# Train a simple model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.bar_chart(importances)

# Model Evaluation
st.subheader("📊 Model Evaluation")
y_pred = model.predict(X_test)
st.text("🔍 Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
st.subheader("🧩 Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Interactive Scatter Plot
st.subheader("📌 Interactive Data Visualization")
if plotly_available:
    x_axis = st.selectbox("Select X-axis feature", df.columns)
    y_axis = st.selectbox("Select Y-axis feature", df.columns)
    try:
        st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis, color="Converted"))
    except Exception as e:
        st.error(f"Error generating Plotly visualization: {e}")
else:
    st.warning("⚠️ Plotly is not installed. Run `pip install plotly` to enable interactive visualizations.")
