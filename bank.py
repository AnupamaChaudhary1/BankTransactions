# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LinearRegression
# from sklearn.cluster import KMeans
# from sklearn.metrics import accuracy_score, mean_squared_error
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# st.title("Bank Transactions Analysis - Simple Output")

# @st.cache_data
# def load_csv(path):
#     df = pd.read_csv(path)
#     df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
#     df = df.dropna(subset=['transaction_amount'])
#     if 'transaction_date' in df.columns:
#         df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
#     return df

# auto_loaded_df = None
# if os.path.exists("cleaned.csv"):
#     auto_loaded_df = load_csv("cleaned.csv")

# uploaded_file = st.file_uploader("Upload cleaned.csv (optional manual upload)")

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
#     df = df.dropna(subset=['transaction_amount'])
#     if 'transaction_date' in df.columns:
#         df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
#     st.success("Data loaded from manual upload")
# elif auto_loaded_df is not None:
#     df = auto_loaded_df
#     st.info("Auto-loaded cleaned.csv from folder")
# else:
#     df = None
#     st.warning("Please upload your cleaned.csv file to continue.")

# if df is not None:
#     st.write("### Data Preview")
#     st.dataframe(df.head())

#     use_case = st.selectbox("Choose Use Case", [
#         "1. Predict High-Spending Customers",
#         "2. Predict Transaction Channel",
#         "3. Forecast Daily Transaction Amount",
#         "4. Detect Customer Segments",
#         "5. Detect Potential Fraud"
#     ])

#     if use_case == "1. Predict High-Spending Customers":
#         st.header("High-Spending Customer Prediction")
#         threshold = df['transaction_amount'].quantile(0.90)
#         df['high_spender'] = (df['transaction_amount'] >= threshold).astype(int)

#         X = pd.get_dummies(df[['transaction_amount', 'account_type', 'transaction_type']], drop_first=True)
#         y = df['high_spender']

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         model = RandomForestClassifier(random_state=42)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         acc = accuracy_score(y_test, y_pred)
#         st.write(f"Model Accuracy: **{acc*100:.2f}%**")
#         st.write(f"Total customers predicted as high spenders: {sum(y_pred)} out of {len(y_pred)} tested.")

#     elif use_case == "2. Predict Transaction Channel":
#         st.header("Transaction Channel Prediction")
#         X = pd.get_dummies(df[['transaction_amount', 'transaction_type', 'account_type']], drop_first=True)
#         y = df['channel']

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         model = DecisionTreeClassifier(random_state=42)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         acc = accuracy_score(y_test, y_pred)
#         st.write(f"Model Accuracy: **{acc*100:.2f}%**")
#         st.write("Sample predictions:")
#         st.dataframe(pd.DataFrame({'Actual': y_test.iloc[:10], 'Predicted': y_pred[:10]}))

#     elif use_case == "3. Forecast Daily Transaction Amount":
#         st.header("Daily Transaction Amount Forecast")
#         daily = df.groupby('transaction_date')['transaction_amount'].sum().reset_index()
#         daily = daily.dropna()
#         daily['day_num'] = np.arange(len(daily))

#         model = LinearRegression()
#         model.fit(daily[['day_num']], daily['transaction_amount'])
#         daily['predicted'] = model.predict(daily[['day_num']])

#         fig, ax = plt.subplots()
#         ax.plot(daily['transaction_date'], daily['transaction_amount'], label='Actual')
#         ax.plot(daily['transaction_date'], daily['predicted'], label='Predicted', linestyle='--')
#         ax.legend()
#         st.pyplot(fig)

#         mse = mean_squared_error(daily['transaction_amount'], daily['predicted'])
#         st.write(f"Mean Squared Error (lower is better): **{mse:.2f}**")

#     elif use_case == "4. Detect Customer Segments":
#         st.header("Customer Segmentation")
#         grouped = df.groupby('customer_id').agg({
#             'transaction_amount': 'mean',
#             'transaction_id': 'count',
#             'channel': pd.Series.nunique
#         }).rename(columns={
#             'transaction_amount': 'avg_amount',
#             'transaction_id': 'txn_count',
#             'channel': 'channels_used'
#         }).reset_index()

#         X = grouped[['avg_amount', 'txn_count', 'channels_used']]
#         model = KMeans(n_clusters=3, random_state=42)
#         grouped['segment'] = model.fit_predict(X)

#         fig, ax = plt.subplots()
#         sns.scatterplot(data=grouped, x='avg_amount', y='txn_count', hue='segment', palette='Set1', ax=ax)
#         ax.set_title('Customer Segments')
#         st.pyplot(fig)

#         st.write("Segments colors correspond to groups of customers with similar spending and transaction patterns.")

#     elif use_case == "5. Detect Potential Fraud":
#         st.header("Potential Fraud Detection")
#         threshold = df['transaction_amount'].quantile(0.99)
#         df['is_fraud'] = (df['transaction_amount'] >= threshold).astype(int)

#         X = pd.get_dummies(df[['transaction_amount', 'transaction_type', 'account_type']], drop_first=True)
#         y = df['is_fraud']

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         model = RandomForestClassifier(random_state=42)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         acc = accuracy_score(y_test, y_pred)
#         st.write(f"Model Accuracy: **{acc*100:.2f}%**")
#         st.write(f"Transactions flagged as potential fraud: {sum(y_pred)} out of {len(y_pred)} tested.")







import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

st.title("Bank Transaction Use Cases with Manual Data Entry")

@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
    df = df.dropna(subset=['transaction_amount'])
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    return df

# Load data
auto_loaded_df = None
if os.path.exists("cleaned.csv"):
    auto_loaded_df = load_csv("cleaned.csv")

uploaded_file = st.file_uploader("Upload cleaned.csv (optional)")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
    df = df.dropna(subset=['transaction_amount'])
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    st.success("✅ Data loaded from manual upload")
elif auto_loaded_df is not None:
    df = auto_loaded_df
    st.info("ℹ️ Auto-loaded cleaned.csv from folder")
else:
    df = None
    st.warning("⚠️ Please upload your cleaned.csv file to start.")

if df is not None:
    st.write("### Data Preview")
    st.dataframe(df.head())

    st.write("---")
    st.header("Add a New Transaction Manually")

    with st.form("manual_entry_form"):
        transaction_id = st.text_input("Transaction ID", "")
        customer_id = st.text_input("Customer ID", "")
        account_type = st.selectbox("Account Type", options=df['account_type'].unique())
        branch_code = st.text_input("Branch Code", "")
        transaction_type = st.selectbox("Transaction Type", options=df['transaction_type'].unique())
        transaction_amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
        transaction_date = st.date_input("Transaction Date", datetime.today())
        channel = st.selectbox("Channel", options=df['channel'].unique())
        balance_after_transaction = st.number_input("Balance After Transaction", format="%.2f")
        merchant_category_code = st.text_input("Merchant Category Code", "")
        geo_location = st.text_input("Geo Location", "")

        submitted = st.form_submit_button("Add Transaction")

    if submitted:
        # Validate required fields
        if not transaction_id or not customer_id:
            st.error("Transaction ID and Customer ID are required.")
        else:
            new_data = {
                'transaction_id': transaction_id,
                'customer_id': customer_id,
                'account_type': account_type,
                'branch_code': branch_code,
                'transaction_type': transaction_type,
                'transaction_amount': transaction_amount,
                'transaction_date': pd.to_datetime(transaction_date),
                'channel': channel,
                'balance_after_transaction': balance_after_transaction,
                'merchant_category_code': merchant_category_code,
                'geo_location': geo_location
            }
            # Append new data to dataframe
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            st.success("✅ New transaction added!")
            st.dataframe(df.tail())

    # Now the use cases run on the updated df with manual data appended

    use_case = st.selectbox("Select Use Case", [
        "1. Predict High-Spending Customers",
        "2. Predict Transaction Channel",
        "3. Forecast Daily Transaction Amount",
        "4. Detect Customer Segments",
        "5. Detect Potential Fraud"
    ])

    if use_case == "1. Predict High-Spending Customers":
        st.header("Predict High-Spending Customers")
        threshold = df['transaction_amount'].quantile(0.90)
        df['high_spender'] = (df['transaction_amount'] >= threshold).astype(int)

        X = pd.get_dummies(df[['transaction_amount', 'account_type', 'transaction_type']], drop_first=True)
        y = df['high_spender']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: **{acc*100:.2f}%**")
        st.write(f"Predicted high spenders: {sum(y_pred)} out of {len(y_pred)} samples.")

    elif use_case == "2. Predict Transaction Channel":
        st.header("Predict Transaction Channel")
        X = pd.get_dummies(df[['transaction_amount', 'transaction_type', 'account_type']], drop_first=True)
        y = df['channel']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: **{acc*100:.2f}%**")
        st.write("Sample predictions:")
        st.dataframe(pd.DataFrame({'Actual': y_test.iloc[:10].values, 'Predicted': y_pred[:10]}))

    elif use_case == "3. Forecast Daily Transaction Amount":
        st.header("Forecast Daily Transaction Amount")
        daily = df.groupby('transaction_date')['transaction_amount'].sum().reset_index()
        daily = daily.dropna()
        daily['day_num'] = np.arange(len(daily))

        model = LinearRegression()
        model.fit(daily[['day_num']], daily['transaction_amount'])
        daily['predicted'] = model.predict(daily[['day_num']])

        fig, ax = plt.subplots()
        ax.plot(daily['transaction_date'], daily['transaction_amount'], label='Actual')
        ax.plot(daily['transaction_date'], daily['predicted'], label='Predicted', linestyle='--')
        ax.legend()
        st.pyplot(fig)

        mse = mean_squared_error(daily['transaction_amount'], daily['predicted'])
        st.write(f"Mean Squared Error: **{mse:.2f}**")

    elif use_case == "4. Detect Customer Segments":
        st.header("Customer Segmentation")
        grouped = df.groupby('customer_id').agg({
            'transaction_amount': 'mean',
            'transaction_id': 'count',
            'channel': pd.Series.nunique
        }).rename(columns={
            'transaction_amount': 'avg_amount',
            'transaction_id': 'txn_count',
            'channel': 'channels_used'
        }).reset_index()

        X = grouped[['avg_amount', 'txn_count', 'channels_used']]
        model = KMeans(n_clusters=3, random_state=42)
        grouped['segment'] = model.fit_predict(X)

        fig, ax = plt.subplots()
        sns.scatterplot(data=grouped, x='avg_amount', y='txn_count', hue='segment', palette='Set1', ax=ax)
        ax.set_title('Customer Segments')
        st.pyplot(fig)

    elif use_case == "5. Detect Potential Fraud":
        st.header("Detect Potential Fraud")
        threshold = df['transaction_amount'].quantile(0.99)
        df['is_fraud'] = (df['transaction_amount'] >= threshold).astype(int)

        X = pd.get_dummies(df[['transaction_amount', 'transaction_type', 'account_type']], drop_first=True)
        y = df['is_fraud']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: **{acc*100:.2f}%**")
        st.write(f"Transactions flagged as potential fraud: {sum(y_pred)} out of {len(y_pred)} tested.")
