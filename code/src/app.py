import pandas as pd
import random
from faker import Faker
import warnings
from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os
import streamlit as st
from langchain.prompts import PromptTemplate

warnings.filterwarnings('ignore')

# Initialize Faker instance
fake = Faker()


# Generate Customer Demographics Data
def generate_customer_demographics(num_customers=1000):
    customer_data = []
    for _ in range(num_customers):
        customer = {
            'customer_id': fake.uuid4(),
            'name': fake.name(),
            'age': random.randint(18, 70),
            'gender': random.choice(['Male', 'Female']),
            'marital_status': random.choice(['Single', 'Married', 'Divorced']),
            'education': random.choice(['High School', 'Bachelor', 'Master', 'PhD']),
            'occupation': fake.job(),
            'salary': random.randint(20000, 150000),  # Yearly salary
        }
        customer_data.append(customer)
    return pd.DataFrame(customer_data)

# Generate Customer Financial Behavior Data
def generate_financial_behavior(customer_ids, num_records=2000):
    financial_data = []
    for _ in range(num_records):
        product_type = random.choice(['Personal Loan', 'Home Loan', 'Credit Card'])
        loan_amount = random.randint(5000, 500000) if product_type != 'Credit Card' else random.randint(5000, 150000)
        credit_limit = random.randint(1000, 150000) if product_type == 'Credit Card' else None
        utilization = random.uniform(0.1, 1.0) if product_type == 'Credit Card' else None
        max_dpd = random.choice([0, 15, 30, 60, 90, 120])
        default_status = random.choice([True, False])

        financial_behavior = {
            'customer_id': random.choice(customer_ids),
            'product_type': product_type,
            'loan_amount': loan_amount,
            'credit_limit': credit_limit,
            'credit_utilization': utilization,
            'emi_paid': random.randint(1, 24),
            'tenure_months': random.randint(12, 60),
            'max_dpd': max_dpd,
            'default_status': default_status
        }
        financial_data.append(financial_behavior)
    return pd.DataFrame(financial_data)

# Generate Customer Enquiries Data (Last 3 months)
def generate_customer_enquiries(customer_ids, num_records=500):
    enquiries_data = []
    for _ in range(num_records):
        product_type = random.choice(['Personal Loan', 'Home Loan', 'Credit Card'])
        enquiry_amount = random.randint(5000, 500000) if product_type != 'Credit Card' else random.randint(5000, 100000)
        enquiry = {
            'customer_id': random.choice(customer_ids),
            'enquiry_date': fake.date_between(start_date='-90d', end_date='today'),
            'product_type': product_type,
            'enquiry_amount': enquiry_amount,
            'status': random.choice(['Approved', 'Rejected'])
        }
        enquiries_data.append(enquiry)
    return pd.DataFrame(enquiries_data)

# Generate Customer Transaction Data (Past 6 months)
def generate_customer_transactions(customer_ids, num_records=5000):
    transactions_data = []
    for _ in range(num_records):
        transaction_date = fake.date_between(start_date='-180d', end_date='today')
        transaction_amount = random.uniform(50, 10000)

        # Transaction description with salary-related and hobby keywords
        transaction_description = random.choice([
            'Salary from XYZ Corp', 'Amazon Purchase', 'Grocery Store', 'Gym Membership',
            'Netflix Subscription', 'Restaurant', 'Fuel Station', 'Travel Booking',
            'SALARY - ABC Corp', 'SAL credited from DEF Ltd', 'Monthly Salary GHI Pvt Ltd',
            'Rent Payment', 'Car Insurance', 'Mobile Phone Bill', 'Electricity Bill', 'Spotify Subscription',
            'Uber Ride', 'Etsy Shopping', 'Concert Ticket', 'Books Purchase'
        ])

        # Salary detection
        salary_keywords = ['Salary', 'SALARY', 'SAL', 'SAL credited', 'Monthly Salary']
        is_salary = any(keyword in transaction_description.upper() for keyword in salary_keywords)

        # Hobbies detection based on transaction descriptions
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

        transaction = {
            'customer_id': random.choice(customer_ids),
            'transaction_date': transaction_date,
            'transaction_amount': transaction_amount,
            'transaction_description': transaction_description,
            'account_balance': random.uniform(500, 20000),
            'is_salary': is_salary,
            'hobby_detected': hobbies
        }
        transactions_data.append(transaction)

    return pd.DataFrame(transactions_data)

def generate_customer_sentiments(customer_ids, num_records=5000):
    sentiments_data = []

    sentiment_sources = ['Twitter', 'Facebook', 'Instagram', 'LinkedIn', 'Reddit', 'TrustPilot', 'Google Reviews', 'YouTube Comments', 'Quora', 'Forums']
    sentiment_labels = ['Positive', 'Neutral', 'Negative']

    product_keywords = [
        'Credit Card', 'Loan', 'Mutual Fund', 'Stock', 'Insurance', 'Netflix', 'Spotify', 'Gym Membership', 'Mortgage', 'Savings Account', 'Investment Plan',
        'Health Insurance', 'Car Loan', 'Home Loan', 'Travel Insurance', 'Mobile Phone Plan', 'Laptop', 'Smartwatch', 'Streaming Service', 'Online Course', 'Luxury Watch',
        'Gaming Console', 'Electric Vehicle', 'Home Security System', 'Smart Home Device', 'E-book Subscription', 'Meal Delivery Service', 'Fitness Tracker', 'Digital Wallet'
    ]

    intent_categories = {
        'Product Interest': ['Looking for suggestions', 'What should I buy?', 'Any recommendations?', 'Best choice for me?', 'Which one is better?', 'Need a new option'],
        'Service Satisfaction': ['Excellent service', 'Great support', 'Fantastic experience', 'Highly recommend', 'Loved my experience', 'Poor service', 'Frustrated', 'Regret', 'Worst experience'],
        'Technical Support': ['Not working', 'Facing issues', 'Bug found', 'App crashes', 'Error message', 'Glitchy experience', 'Feature broken'],
        'Financial Concern': ['Unexpected charges', 'Hidden fees', 'Interest rates too high', 'Account frozen', 'Fraudulent transaction', 'Unauthorized deduction', 'Late fee issue', 'Credit score impact'],
        'Investment Interest': ['Best savings account', 'High-interest deposit', 'Mutual fund recommendations', 'Stock investment tips', 'Retirement planning', 'Cryptocurrency advice', 'Is this a good investment?'],
        'Loan & Credit Inquiry': ['Loan eligibility', 'Credit card approval', 'Best mortgage rates', 'Personal loan options', 'Debt consolidation', 'EMI calculation', 'Low-interest credit card'],
        'Subscription Inquiry': ['Netflix subscription', 'Spotify plan', 'Gym membership', 'Service renewal', 'Want to upgrade', 'Cancel subscription'],
        'Customer Support': ['Need assistance', 'Support not responding', 'How do I contact?', 'Live chat not available', 'Waiting for response'],
        'Comparison': ['Better than', 'Worse than', 'Compared to', 'Alternative to', 'How does this compare?', 'Which is best?'],
        'Refund Request': ['Want my money back', 'Need a refund', 'Did not like it', 'Return process', 'Refund issue', 'Money not credited']
    }

    for _ in range(num_records):
        sentiment_source = random.choice(sentiment_sources)
        sentiment_label = random.choice(sentiment_labels)
        sentiment_score = {'Positive': random.uniform(0.6, 1.0), 'Neutral': random.uniform(0.4, 0.6), 'Negative': random.uniform(0.0, 0.4)}[sentiment_label]

        # Assign product dynamically
        product_mentioned = random.choice(product_keywords)

        # Generate sentiment text including product
        sentiment_text = f"{random.choice(intent_categories['Product Interest'])} about {product_mentioned}" if sentiment_label != 'Negative' else f"{random.choice(intent_categories['Service Satisfaction'])} with {product_mentioned}"

        # Determine intent based on sentiment text
        intent = next((key for key, values in intent_categories.items() if any(phrase in sentiment_text for phrase in values)), 'Product Interest')

        sentiment_entry = {
            'customer_id': random.choice(customer_ids),
            'sentiment_date': fake.date_between(start_date='-180d', end_date='today'),
            'sentiment_source': sentiment_source,
            'sentiment_text': sentiment_text,
            'sentiment_label': sentiment_label,
            'sentiment_score': round(sentiment_score, 2),
            'intent': intent,
            'product_mentioned': product_mentioned
        }
        sentiments_data.append(sentiment_entry)

    return pd.DataFrame(sentiments_data)


def generate_customer_data(num_customers=5000, num_financial_records=15000, num_enquiries=4000, num_transactions=20000, num_sentiments=6000):
    """Generates and aggregates customer data for personalization and recommendation."""

    # Generate Data
    customers = generate_customer_demographics(num_customers)
    financial_behavior = generate_financial_behavior(customers['customer_id'], num_records=num_financial_records)
    enquiries = generate_customer_enquiries(customers['customer_id'], num_records=num_enquiries)
    transactions = generate_customer_transactions(customers['customer_id'], num_records=num_transactions)
    social_sentiments = generate_customer_sentiments(customers['customer_id'], num_records=num_sentiments)
    print(social_sentiments.head())
    # Financial Summary
    financial_summary = financial_behavior.groupby('customer_id').agg({
        'loan_amount': 'mean',
        'credit_limit': 'mean',
        'credit_utilization': 'mean',
        'emi_paid': 'sum',
        'tenure_months': 'mean',
        'max_dpd': 'max',
        'default_status': 'mean',
        'product_type': lambda x: list(x.unique())  # Convert to list for readability
    }).reset_index()

    # Transaction Summary (Fixing the `is_salary` filter issue)
    transaction_summary = transactions.groupby('customer_id').agg({
        'transaction_amount': 'mean',
        'account_balance': 'mean',
        'is_salary': 'sum'
    }).reset_index()

    # Ensure 'is_salary' exists before filtering
    if 'is_salary' in transactions.columns:
        salary_transactions = transactions[transactions['is_salary'] == 1]
        salary_summary = salary_transactions.groupby('customer_id')['transaction_amount'].sum().reset_index()
        salary_summary.rename(columns={'transaction_amount': 'total_salary_received'}, inplace=True)
        transaction_summary = pd.merge(transaction_summary, salary_summary, on='customer_id', how='left')

    # Enquiries Summary (Fixing column name consistency)
    enquiries_summary = enquiries.groupby('customer_id').agg({
        'enquiry_amount': 'mean',
        'product_type': lambda x: x.nunique(),
        'customer_id': 'count'
    }).rename(columns={
        'customer_id': 'total_enquiries',
        'product_type': 'unique_products_enquired'
    }).reset_index()

    # Sentiment Summary
    sentiment_summary = social_sentiments.groupby('customer_id').agg({
        'sentiment_score': 'mean',
        'intent': lambda x: x.mode()[0] if not x.mode().empty else 'General',
        'sentiment_source': 'first',
        'sentiment_text': 'first',
        'sentiment_label': 'first',
        'product_mentioned': 'first'
    }).reset_index()

    # Merge All Data
    merged_data = pd.merge(customers, financial_summary, on='customer_id', how='left')
    merged_data = pd.merge(merged_data, enquiries_summary, on='customer_id', how='left')
    merged_data = pd.merge(merged_data, transaction_summary, on='customer_id', how='left')
    merged_data = pd.merge(merged_data, sentiment_summary, on='customer_id', how='left')

    # Step 1: Explode the list in 'product_type' column
    df_exploded = merged_data.explode('product_type')
    # Step 2: One-hot encode the 'product_type' column
    df_encoded = pd.get_dummies(df_exploded['product_type'])
    merged_data = pd.concat([df_exploded, df_encoded], axis=1)
    # Step 4: Group by the original index and aggregate to bring it back into one row per customer
    df_final = merged_data.groupby(merged_data.index).sum()

    # Define the aggregation function for each column
    aggregation_functions = {
    'customer_id': 'first',  # Keep first occurrence (assuming it's the same for the group)
    'name': 'first',         # Keep the first name in each group
    'age': 'mean',           # For age, you can take the average or median
    'gender': 'first',       # Assuming gender is the same within each group, take the first
    'marital_status': 'first', # Same for marital status
    'education': 'first',    # Same for education
    'occupation': 'first',   # Same for occupation
    'salary': 'sum',         # Sum numerical values like salary
    'loan_amount': 'sum',    # Sum numerical values like loan amount
    'credit_limit': 'sum',   # Sum numerical values like credit limit
    'credit_utilization': 'sum',
    'emi_paid':'sum',
    'tenure_months':'sum',
    'max_dpd':'max',
    'default_status':'max',
    'enquiry_amount': 'sum',
    'unique_products_enquired': 'sum',
    'total_enquiries': 'sum',
    'transaction_amount': 'sum',
    'account_balance': 'sum',
    'is_salary': 'mean',     # For boolean-like columns, you can take the mean (0 or 1)
    'sentiment_source': 'first',
    'sentiment_text': 'first',
    'sentiment_label': 'first',
    'sentiment_score': 'mean',
    'product_mentioned': 'first',
    'intent': 'first',
    'Credit Card': 'max',    # For categorical (binary) features, take max (0 or 1)
    'Home Loan': 'max',
    'Personal Loan': 'max',
    }

    
    print(merged_data.groupby(merged_data.index))
    # Group by and apply aggregation functions
    df_final = merged_data.groupby(merged_data.index).agg(aggregation_functions)
    df_final['content'] = df_final.apply(lambda row: f"Based on the following customer data: {row.to_dict()}, suggest suitable products.", axis=1)

    return df_final


def create_chroma_vector_store(df, persist_directory='chroma_data'):
    documents = [Document(page_content=row['content'], metadata={'class': row['age']}) for _, row in df.iterrows()]
    hg_embeddings = HuggingFaceEmbeddings()
    langchain_chroma = Chroma.from_documents(
        documents=documents,
        collection_name="recommendation_engine",
        embedding=hg_embeddings,
        persist_directory=persist_directory
    )
    return langchain_chroma


def initialize_model(model_id='HuggingFaceH4/zephyr-7b-beta'):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    model_config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True, max_new_tokens=1024)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        max_length=6000,
        max_new_tokens=500,
        device_map="auto",
    )
    return HuggingFacePipeline(pipeline=query_pipeline)


import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ---- Set Page Configuration ----
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="💰",
    layout="wide"
)

# ---- Custom CSS for Better Styling ----
st.markdown("""
    <style>
        /* Background color */
        body, [data-testid="stAppViewContainer"] {
            background-color: #F4F4F4;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #2C3E50;
            color: white;
        }

        /* Sidebar text color */
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p {
            color: white !important;
        }

        /* Input box styling */
        input[type="text"] {
            border-radius: 10px;
            padding: 10px;
            border: 2px solid #3498DB;
            font-size: 16px;
        }

        /* Button styling */
        div.stButton > button {
            background-color: #3498DB;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #2980B9;
        }

        /* Styled recommendation box */
        .result-box {
            border: 2px solid #3498DB;
            border-radius: 10px;
            padding: 15px;
            background-color: white;
            font-size: 18px;
        }

    </style>
""", unsafe_allow_html=True)


def main():
    # ---- Sidebar for Branding & Navigation ----
    with st.sidebar:
        st.image("/content/logo-png.png", use_container_width=True)  # Full-width logo
        st.markdown("<h2 style='text-align: center;'>AI Financial Advisor</h2>", unsafe_allow_html=True)
        st.write("💡 Get personalized recommendations based on your profile.")
        st.write("🔹 Enter Your user details.")
        st.write("🔹 Click **Get Recommendation** to see suggestions.")
        st.markdown("---")  # Adds a separator
    
    # ---- Main App UI ----
    st.markdown("<h1 style='text-align: center; color: #2C3E50;'>💰 AI-Powered Personalized Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #555;'>Enter Your user details:</h3>", unsafe_allow_html=True)
    
    # Centered Input Box
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        user_input = st.text_input("Enter your query:", placeholder='e.g., {"customer_id": "a85af6e6-0033-4998-a0f4-5e8519896552", "name": "Jeffrey Nicholson", "age": 55.0, "gender": "Male", "marital_status": "Divorced", "education": "Bachelor", "occupation": "Medical sales representative", "salary": 79520, "loan_amount": 673258.6666666666, "credit_limit": 48706.0, "credit_utilization": 0.3406952638277163, "emi_paid": 84.0, "tenure_months": 60.666666666666664, "max_dpd": 120.0, "default_status": 0.3333333333333333, "enquiry_amount": 285356.0, "unique_products_enquired": 2.0, "total_enquiries": 2.0, "transaction_amount": 8657.701868940421, "account_balance": 19737.778617612592, "is_salary": 1.0, "Credit Card": true, "Home Loan": true, "Personal Loan": false}', label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    
    # Centered Button
    col4, col5, col6 = st.columns([2, 2, 2])
    with col5:
        submit_button = st.button("🔍 Get Recommendation", use_container_width=True)
    
    # ---- Handle Recommendation Logic ----
    if submit_button:
        if user_input:
            with st.spinner("🔄 Processing your request..."):
                try:
                    # Load data and models
                    df_final = generate_customer_data()
                    langchain_chroma = create_chroma_vector_store(df_final)
                    llm = initialize_model()
                    
                    # Define prompt
                    template = """
                    Based on the following customer data, suggest a suitable financial product.
                    Return only the product name with a brief explanation.
                    Customer Data: {question}
                    Context: {context}
                    Answer:
                    """
                    PROMPT = PromptTemplate(template=template, input_variables=["context","question"])
                    
                    # Set up QA Chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=langchain_chroma.as_retriever(search_kwargs={"k": 1}),
                        chain_type_kwargs={"prompt": PROMPT}
                    )
                    
                    # Generate response
                    response = qa_chain({"query": user_input})
                    
                    # ---- Display Styled Response ----
                    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                    st.success("✅ Recommendation:")
                    st.markdown(f"""
                    <div class="result-box">
                        <b>{response["result"]}</b>
                    </div>
                    """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
        else:
            st.warning("⚠️ Please enter a query.")

if __name__ == "__main__":
    main()

