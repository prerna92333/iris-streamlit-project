import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Page configuration
st.set_page_config(page_title="Interactive Iris Dashboard", layout="wide")

# Load data from the SQLite database
@st.cache_data
def load_data():
    conn = sqlite3.connect("iris_extended_db.sqlite")
    df = pd.read_sql("SELECT * FROM IrisMain", conn)
    conn.close()
    return df

# Function to train and predict species using KNN
def train_knn(df):
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

# Load data
df = load_data()

# Sidebar Components
st.sidebar.title("Interactive Filters")

# Dropdown to select X and Y axes for scatter plots
x_axis = st.sidebar.selectbox("Select X-axis", df.columns[3:7])
y_axis = st.sidebar.selectbox("Select Y-axis", df.columns[3:7])

# Sliders for filtering ranges
sepal_length_range = st.sidebar.slider("Sepal Length Range (cm)", 4.0, 8.0, (4.0, 8.0))
petal_length_range = st.sidebar.slider("Petal Length Range (cm)", 1.0, 7.0, (1.0, 7.0))

# Reset button
if st.sidebar.button("Reset Filters"):
    sepal_length_range = (4.0, 8.0)
    petal_length_range = (1.0, 7.0)

# Filtered data
filtered_df = df[
    (df['sepal_length'].between(sepal_length_range[0], sepal_length_range[1])) &
    (df['petal_length'].between(petal_length_range[0], petal_length_range[1]))
]

# Main Layout
st.title("Interactive Iris Data Analysis Dashboard")
st.write("Explore and interact with the Iris dataset dynamically.")

# Display filtered data
st.write("### Filtered Dataset")
st.dataframe(filtered_df)

# KPIs Section
st.write("### Key Performance Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", len(filtered_df))
col2.metric("Avg Sepal Length", round(filtered_df["sepal_length"].mean(), 2))
col3.metric("Avg Petal Length", round(filtered_df["petal_length"].mean(), 2))

# Visualization Section
st.write("### Dynamic Visualizations")

# Scatter Plot
st.write("#### Scatter Plot")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, hue="species", ax=ax)
plt.title(f"{y_axis} vs {x_axis}")
st.pyplot(fig)

# Histogram
st.write("#### Histogram of Sepal Length")
fig, ax = plt.subplots()
sns.histplot(filtered_df['sepal_length'], kde=True, color='skyblue', bins=20, ax=ax)
st.pyplot(fig)

# Average Metrics by Species
st.write("#### Average Sepal and Petal Length by Species")
avg_metrics = filtered_df.groupby("species")[["sepal_length", "petal_length"]].mean()
st.bar_chart(avg_metrics)

# Prediction Section
st.write("### KNN Species Prediction")
model, accuracy = train_knn(df)
st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")

# Allow users to input values for prediction
st.write("#### Input Values for Prediction")
sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.5)
petal_width = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.5)

# Make Prediction
if st.button("Predict Species"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.success(f"The predicted species is: **{prediction[0]}**")

# Insights Section
st.write("### Key Insights")
st.markdown("""
- Use the **dropdowns** to dynamically select variables for scatter plots.
- Adjust the sliders to filter Sepal and Petal Length ranges.
- Observe the distribution of Sepal Length using the histogram.
- Train a **K-Nearest Neighbors (KNN)** model for predicting Iris species.
- Use the **Prediction Section** to test new inputs and get real-time predictions.
""")

