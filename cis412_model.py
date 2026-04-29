import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

st.set_page_config(page_title="Airline Customer Satisfaction Dashboard", layout="wide")
st.title("✈️ Airline Customer Satisfaction Dashboard")


@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    def drop_extra_index(df):
        first_col = str(df.columns[0])
        if first_col.startswith("Unnamed") or first_col.lower() in {"id", "index"}:
            return df.iloc[:, 1:].copy()
        return df.copy()

    train = drop_extra_index(train)
    test = drop_extra_index(test)
    return train, test


train, test = load_data()

if "satisfaction" not in train.columns or "satisfaction" not in test.columns:
    st.error("The column 'satisfaction' must exist in both train.csv and test.csv.")
    st.stop()

st.sidebar.header("Model Settings")
max_depth = st.sidebar.slider("Decision Tree max depth", 1, 20, 7)

st.subheader("Data Preview")
col1, col2 = st.columns(2)
with col1:
    st.write("Train Data")
    st.dataframe(train.head())
with col2:
    st.write("Test Data")
    st.dataframe(test.head())

# Split features and target
X_train = train.drop(columns=["satisfaction"])
y_train = train["satisfaction"]

X_test = test.drop(columns=["satisfaction"])
y_test = test["satisfaction"]

# Encode target
y_train = y_train.map({"satisfied": 1, "neutral or dissatisfied": 0})
y_test = y_test.map({"satisfied": 1, "neutral or dissatisfied": 0})

# Categorical columns
cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
cat_cols = [c for c in cat_cols if c in X_train.columns]

# One-hot encode categorical variables
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
preprocessor = ColumnTransformer(
    transformers=[("cat", one_hot_encoder, cat_cols)],
    remainder="passthrough"
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

encoded_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
other_features = [col for col in X_train.columns if col not in cat_cols]
all_feature_names = list(encoded_feature_names) + other_features

X_train_encoded = pd.DataFrame(X_train_encoded, columns=all_feature_names, index=X_train.index)
X_test_encoded = pd.DataFrame(X_test_encoded, columns=all_feature_names, index=X_test.index)

st.subheader("Model Training")
model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=0,
    max_depth=max_depth
)
model.fit(X_train_encoded, y_train)

y_pred = model.predict(X_test_encoded)
y_prob = model.predict_proba(X_test_encoded)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{accuracy:.3f}")
m2.metric("Precision", f"{precision:.3f}")
m3.metric("Recall", f"{recall:.3f}")
m4.metric("F1 Score", f"{f1:.3f}")
m5.metric("AUC", f"{auc_score:.3f}")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
ax.set_title("Confusion Matrix - Test Set")
st.pyplot(fig)

st.subheader("ROC Curve")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

st.subheader("Feature Importance")
feature_importances = pd.DataFrame({
    "Feature": X_train_encoded.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feature_importances, x="Importance", y="Feature", ax=ax)
ax.set_title("Top Feature Importances")
st.pyplot(fig)

st.subheader("Satisfaction by Class")
satisfaction_by_class = train.groupby(["Class", "satisfaction"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 5))
satisfaction_by_class.plot(kind="bar", stacked=True, ax=ax, color=["lightcoral", "lightskyblue"])
ax.set_xlabel("Class")
ax.set_ylabel("Number of Customers")
ax.set_title("Satisfaction Distribution by Class")
ax.legend(title="Satisfaction")
st.pyplot(fig)

st.subheader("Satisfaction by Customer Type")
satisfaction_by_customer_type = train.groupby(["Customer Type", "satisfaction"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 5))
satisfaction_by_customer_type.plot(kind="bar", stacked=True, ax=ax, color=["lightcoral", "lightskyblue"])
ax.set_xlabel("Customer Type")
ax.set_ylabel("Number of Customers")
ax.set_title("Satisfaction Distribution by Customer Type")
ax.legend(title="Satisfaction")
st.pyplot(fig)

st.header("🔮 Predict Customer Satisfaction")

st.write("Enter passenger details to predict satisfaction:")

# --- User Inputs ---
gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])

age = st.slider("Age", 10, 80, 30)
flight_distance = st.slider("Flight Distance", 0, 5000, 1000)

wifi = st.slider("Inflight wifi service (0-5)", 0, 5, 3)
online_boarding = st.slider("Online boarding (0-5)", 0, 5, 3)
seat_comfort = st.slider("Seat comfort (0-5)", 0, 5, 3)
inflight_entertainment = st.slider("Inflight entertainment (0-5)", 0, 5, 3)

# --- Create input dataframe ---
input_dict = {
    "Gender": gender,
    "Customer Type": customer_type,
    "Type of Travel": travel_type,
    "Class": flight_class,
    "Age": age,
    "Flight Distance": flight_distance,
    "Inflight wifi service": wifi,
    "Online boarding": online_boarding,
    "Seat comfort": seat_comfort,
    "Inflight entertainment": inflight_entertainment
}

input_df = pd.DataFrame([input_dict])

# --- Apply same preprocessing ---
input_encoded = preprocessor.transform(input_df)
input_encoded = pd.DataFrame(input_encoded, columns=all_feature_names)

# --- Prediction ---
if st.button("Predict Satisfaction"):
    prediction = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.success(f"✅ Passenger is likely SATISFIED ({prob:.2%} confidence)")
    else:
        st.error(f"❌ Passenger is likely NOT satisfied ({1-prob:.2%} confidence)")
