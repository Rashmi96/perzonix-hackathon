import streamlit as st
import pandas as pd
import random
from faker import Faker
import time

# Initialize Faker
fake = Faker()

# 🎨 Streamlit UI Config
st.set_page_config(page_title="📊 Hyper-personalization and Recommendation", layout="wide")

# 🌟 Custom CSS Styling
st.markdown("""
    <style>
        /* Sidebar Styling */
        .stSidebar {
            background-color: #f4f4f4;
            padding: 15px;
        }

        /* Custom Buttons */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 8px 20px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }

        /* Custom Headers */
        .stMarkdown h2 {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }

        /* Table Styling */
        .stDataFrame {
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# 🌟 App Header
st.markdown("<p style='text-align: center; color: #4CAF50; font-size: 35px; font-weight: bold;'>📊 Hyper-personalization & Recommendation Model Training</p>", unsafe_allow_html=True)
st.markdown("### 🚀 Upload your customer data to train the LLMs!")

# 📂 Sidebar File Upload Section
st.sidebar.header("📂 Upload Files")

customer_file = st.sidebar.file_uploader("📂 Upload Customer Info (CSV, Excel, JSON) (Mandatory)",
                                         type=["csv", "xlsx", "json"], key="customer")
financial_file = st.sidebar.file_uploader("📂 Upload Financial Data (Optional)",
                                          type=["csv", "xlsx", "json"], key="financial")
transactions_file = st.sidebar.file_uploader("📂 Upload Transactions (Optional)",
                                             type=["csv", "xlsx", "json"], key="transactions")
enquiry_file = st.sidebar.file_uploader("📂 Upload Enquiry Data (Optional)",
                                        type=["csv", "xlsx", "json"], key="enquiry")

#  Function to Load Data Based on File Type
def load_data(file):
    if file is None:
        return None  # Return None if no file is uploaded

    file_extension = file.name.split(".")[-1].lower()

    if file_extension == "csv":
        return pd.read_csv(file)
    elif file_extension == "xlsx":
        return pd.read_excel(file)
    elif file_extension == "json":
        return pd.read_json(file)
    else:
        return None  # Unsupported file type

#  Function to Handle Missing Data in Customer Info
def handle_missing_customer_data(df):
    mandatory_columns = ["customer_id", "name", "age", "gender", "marital_status", "education", "occupation", "salary"]
    missing_columns = [col for col in mandatory_columns if col not in df.columns]

    if missing_columns:
        st.warning(f"⚠️ Missing columns: {missing_columns}. Filling missing values with 'Unknown' or default values.")

    if "customer_id" not in df.columns:
        st.error("❌ Error: 'customer_id' column is missing! Please upload a valid file.")
        st.stop()  # Stop execution immediately

    #  Fill missing columns with default values
    for col in mandatory_columns:
        if col not in df.columns or df[col].isnull().all():  # Check if the column is missing or all values are null
            if col == "customer_id":
                df[col] = [fake.uuid4() for _ in range(len(df))]
            elif col == "name":
                df[col] = [fake.name() for _ in range(len(df))]
            elif col == "gender":
                df[col] = [random.choice(['Male', 'Female']) for _ in range(len(df))]
            elif col == "marital_status":
                df[col] = [random.choice(['Single', 'Married', 'Divorced']) for _ in range(len(df))]
            elif col == "age":
                df[col] = [random.randint(18, 70) for _ in range(len(df))]
            elif col == "education":
                df[col] = [random.choice(['High School', 'Bachelor', 'Master', 'PhD']) for _ in range(len(df))]
            elif col == "occupation":
                df[col] = [fake.job() for _ in range(len(df))]
            elif col == "salary":
                df[col] = [random.randint(20000, 150000) for _ in range(len(df))]

    return df[mandatory_columns]  # Ensure correct column order


def handle_missing_financial_data(df):
    financial_mandatory_columns = ["customer_id", "product_type", "loan_amount", "credit_limit", "credit_utilization",
                                   "emi_paid", "tenure_months", "max_dpd", "default_status"]
    missing_columns = [col for col in financial_mandatory_columns if col not in df.columns]

    if missing_columns:
        st.warning(f"⚠️ Missing columns in Financial Data: {missing_columns}. Filling missing values.")

    if "customer_id" not in df.columns:
        st.error("❌ Error: 'customer_id' column is missing! Please upload a valid file.")
        st.stop()  # Stop execution immediately

    for index, row in df.iterrows():
        product_type = row.get("product_type", random.choice(['Personal Loan', 'Home Loan', 'Credit Card']))

        df.at[index, "product_type"] = product_type
        df.at[index, "loan_amount"] = row.get("loan_amount", random.randint(5000,
                                                                            500000) if product_type != 'Credit Card' else random.randint(
            5000, 150000))
        df.at[index, "credit_limit"] = row.get("credit_limit",
                                               random.randint(1000, 150000) if product_type == 'Credit Card' else None)
        df.at[index, "credit_utilization"] = row.get("credit_utilization", random.uniform(0.1,
                                                                                          1.0) if product_type == 'Credit Card' else None)
        df.at[index, "emi_paid"] = row.get("emi_paid", random.randint(1, 24))
        df.at[index, "tenure_months"] = row.get("tenure_months", random.randint(12, 60))
        df.at[index, "max_dpd"] = row.get("max_dpd", random.choice([0, 15, 30, 60, 90, 120]))
        df.at[index, "default_status"] = row.get("default_status", random.choice([True, False]))

    return df[financial_mandatory_columns]  # Ensure correct column order

def handle_missing_enquiry_data(df):
    enquiry_mandatory_columns = ["customer_id", "enquiry_date", "product_type", "enquiry_amount", "status"]
    missing_columns = [col for col in enquiry_mandatory_columns if col not in df.columns]

    if missing_columns:
        st.warning(f"⚠️ Missing columns in Enquiry Data: {missing_columns}. Filling missing values.")

    if "customer_id" not in df.columns:
        st.error("❌ Error: 'customer_id' column is missing! Please upload a valid file.")
        st.stop()  # Stop execution immediately

    for index, row in df.iterrows():
        df.at[index, "enquiry_date"] = row.get("enquiry_date", fake.date_between(start_date='-90d', end_date='today'))
        df.at[index, "product_type"] = row.get("product_type", random.choice(['Personal Loan', 'Home Loan', 'Credit Card']))
        df.at[index, "enquiry_amount"] = row.get("enquiry_amount", random.randint(5000, 500000))
        df.at[index, "status"] = row.get("status", random.choice(['Approved', 'Rejected']))

    return df[enquiry_mandatory_columns]  # Ensure correct column order

def handle_missing_transaction_data(df):
    transaction_mandatory_columns = ["customer_id", "transaction_date", "transaction_amount", "transaction_description",
                                     "account_balance", "is_salary", "hobby_detected"]
    missing_columns = [col for col in transaction_mandatory_columns if col not in df.columns]

    if missing_columns:
        st.warning(f"⚠️ Missing columns in Transaction Data: {missing_columns}. Filling missing values.")

    if "customer_id" not in df.columns:
        st.error("❌ Error: 'customer_id' column is missing! Please upload a valid file.")
        st.stop()  # Stop execution immediately

    for index, row in df.iterrows():
        transaction_description = row.get("transaction_description", random.choice([
            'Salary from XYZ Corp', 'Amazon Purchase', 'Grocery Store', 'Gym Membership',
            'Netflix Subscription', 'Restaurant', 'Fuel Station', 'Travel Booking',
            'SALARY - ABC Corp', 'SAL credited from DEF Ltd', 'Monthly Salary GHI Pvt Ltd',
            'Rent Payment', 'Car Insurance', 'Mobile Phone Bill', 'Electricity Bill', 'Spotify Subscription',
            'Uber Ride', 'Etsy Shopping', 'Concert Ticket', 'Books Purchase'
        ]))

        salary_keywords = ['Salary', 'SALARY', 'SAL', 'SAL credited', 'Monthly Salary']
        is_salary = any(keyword in transaction_description.upper() for keyword in salary_keywords)

        hobbies = None
        if "Amazon" in transaction_description or "Etsy" in transaction_description:
            hobbies = 'Shopping'
        elif "Netflix" in transaction_description or "Spotify" in transaction_description:
            hobbies = 'Entertainment'
        elif "Gym" in transaction_description:
            hobbies = 'Fitness'
        elif "Concert" in transaction_description:
            hobbies = 'Music'
        elif "Books" in transaction_description:
            hobbies = 'Reading'
        elif "Travel" in transaction_description or "Uber Ride" in transaction_description:
            hobbies = 'Travel'

        df.at[index, "transaction_date"] = row.get("transaction_date", fake.date_between(start_date='-180d', end_date='today'))
        df.at[index, "transaction_amount"] = row.get("transaction_amount", random.uniform(50, 10000))
        df.at[index, "transaction_description"] = transaction_description
        df.at[index, "account_balance"] = row.get("account_balance", random.uniform(500, 20000))
        df.at[index, "is_salary"] = is_salary
        df.at[index, "hobby_detected"] = hobbies

    return df[transaction_mandatory_columns]

#  Generate Default Values for Customer Data
def generate_customer_values(col, size):
    if col == "customer_id":
        return [fake.uuid4() for _ in range(size)]
    elif col == "name":
        return [fake.name() for _ in range(size)]
    elif col == "gender":
        return [random.choice(['Male', 'Female']) for _ in range(size)]
    elif col == "marital_status":
        return [random.choice(['Single', 'Married', 'Divorced']) for _ in range(size)]
    elif col == "age":
        return [random.randint(18, 70) for _ in range(size)]
    elif col == "education":
        return [random.choice(['High School', 'Bachelor', 'Master', 'PhD']) for _ in range(size)]
    elif col == "occupation":
        return [fake.job() for _ in range(size)]
    elif col == "salary":
        return [random.randint(20000, 150000) for _ in range(size)]
    return ["Unknown"] * size

#  Function to Process Uploaded Files
def process_uploaded_files(customer_file, financial_file, transactions_file, enquiry_file):
    """Ensures data processing only happens after file uploads and merges all datasets."""

    #  Ensure the Customer Info file is uploaded (Mandatory)
    if customer_file is None:
        st.error("❌ Customer Info is mandatory! Please upload a valid file.")
        st.stop()  # Stop execution

    #  Load Data
    customer_data = load_data(customer_file)
    customer_data = handle_missing_customer_data(customer_data)
    st.markdown("<h4 style='color: #4CAF50; font-size: 18px;'>👤 Customer Data (Processed)</h4>", unsafe_allow_html=True)
    st.dataframe(customer_data)

    #  Load Optional Data
    financial_data = load_data(financial_file) if financial_file else None
    transactions_data = load_data(transactions_file) if transactions_file else None
    enquiry_data = load_data(enquiry_file) if enquiry_file else None

    #  Handle Missing Data for Optional Files
    if financial_data is not None:
        financial_data = handle_missing_financial_data(financial_data)
        st.markdown("<h4 style='color: #4CAF50; font-size: 18px;'>💰 Financial Data (Processed)</h4>", unsafe_allow_html=True)
        st.subheader("💰 Financial Data (Processed)")
        st.dataframe(financial_data)

    if transactions_data is not None:
        transactions_data = handle_missing_transaction_data(transactions_data)
        st.markdown("<h4 style='color: #4CAF50; font-size: 18px;'>💳 Transactions Data (Processed)</h4>", unsafe_allow_html=True)
        st.subheader("💳 Transactions Data (Processed)")
        st.dataframe(transactions_data)

    if enquiry_data is not None:
        enquiry_data = handle_missing_enquiry_data(enquiry_data)
        st.markdown("<h4 style='color: #4CAF50; font-size: 18px;'>📢 Enquiry Data (Processed)</h4>", unsafe_allow_html=True)
        st.subheader("📢 Enquiry Data (Processed)")
        st.dataframe(enquiry_data)

    #  Ensure `merged_data` Always Exists
    merged_data = customer_data.copy()

    #  Merge Only If Data Exists
    if financial_data is not None:
        merged_data = pd.merge(merged_data, financial_data, on="customer_id", how="left")

    if transactions_data is not None:
        merged_data = pd.merge(merged_data, transactions_data, on="customer_id", how="left")

    if enquiry_data is not None:
        merged_data = pd.merge(merged_data, enquiry_data, on="customer_id", how="left")

    #  Check if `merged_data` Exists Before Further Processing
    if merged_data.empty:
        st.warning("⚠️ No data available after merging. Please upload valid files.")
        return

    #  Prevent `.explode()` Errors if `product_type` is Missing
    if 'product_type' in merged_data.columns:
        df_exploded = merged_data.explode('product_type')
        df_encoded = pd.get_dummies(df_exploded['product_type'])
        merged_data = pd.concat([df_exploded, df_encoded], axis=1)

    # Define the aggregation function for each column
    aggregation_functions = {
        'customer_id': 'first',  # Keep first occurrence (assuming it's the same for the group)
        'name': 'first',  # Keep the first name in each group
        'age': 'mean',  # For age, you can take the average or median
        'gender': 'first',  # Assuming gender is the same within each group, take the first
        'marital_status': 'first',  # Same for marital status
        'education': 'first',  # Same for education
        'occupation': 'first',  # Same for occupation
        'salary': 'sum',  # Sum numerical values like salary
        'loan_amount': 'sum',  # Sum numerical values like loan amount
        'credit_limit': 'sum',  # Sum numerical values like credit limit
        'credit_utilization': 'sum',
        'emi_paid': 'sum',
        'tenure_months': 'sum',
        'max_dpd': 'max',
        'default_status': 'max',
        'enquiry_amount': 'sum',
        'unique_products_enquired': 'sum',
        'total_enquiries': 'sum',
        'transaction_amount': 'sum',
        'account_balance': 'sum',
        'is_salary': 'mean',  # For boolean-like columns, you can take the mean (0 or 1)
        'Credit Card': 'max',  # For categorical (binary) features, take max (0 or 1)
        'Home Loan': 'max',
        'Personal Loan': 'max',
    }

    # df_final = merged_data.groupby(merged_data.index).agg(aggregation_functions)
    # df_final['content'] = df_final.apply(
    #     lambda row: f"Based on the following customer data: {row.to_dict()}, suggest suitable products.", axis=1)
    df_final = merged_data

    #  Show Final Merged Data
    st.markdown("<h4 style='color: #4CAF50; font-size: 18px;'>📊 Merged Data</h4>", unsafe_allow_html=True)
    st.dataframe(merged_data)

   # 📌 Initialize Training State in Session
    if "training_started" not in st.session_state:
        st.session_state.training_started = False

    # 🔘 Move "Train the Model" Button **Below** Merged Data
    st.markdown("---")  # UI Separator

    if st.button("🚀 Train the Model"):
        st.session_state.training_started = True  # Update state to start training

    #  Only Start Training if Button is Clicked
    if st.session_state.training_started:
        st.write("🟢 Training in progress...")

        # 🔄 Progress Bar
        progress_bar = st.progress(0)
        status_text = st.empty()  # Placeholder for status updates

        for percent_complete in range(0, 101, 10):  
            time.sleep(1)  # Simulate training time
            progress_bar.progress(percent_complete)
            status_text.text(f"Training Progress: {percent_complete}%")

        #  Training Completed
        status_text.text(" Training Completed!")
        st.success("Model training is complete! Ready for predictions.")


    #  Provide Download Option
    csv = merged_data.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("📥 Download Merged Data", data=csv, file_name="merged_data.csv", mime="text/csv")

#  Run the Processing Function
process_uploaded_files(customer_file, financial_file, transactions_file, enquiry_file)
