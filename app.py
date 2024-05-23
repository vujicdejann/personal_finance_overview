import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Date, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import plotly.express as px
import bcrypt
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load environment variables
load_dotenv()
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
SECRET_KEY = os.getenv('SECRET_KEY')

DATABASE_URL = f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Database models
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    transactions = relationship('Transaction', back_populates='user')
    budget_goals = relationship('BudgetGoal', back_populates='user')

class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    date = Column(Date, nullable=False)
    category = Column(String(50), nullable=False)
    amount = Column(Float, nullable=False)
    description = Column(String(255))
    user = relationship('User', back_populates='transactions')

class BudgetGoal(Base):
    __tablename__ = 'budget_goals'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    category = Column(String(50), nullable=False)
    goal_amount = Column(Float, nullable=False)
    user = relationship('User', back_populates='budget_goals')

Base.metadata.create_all(engine)

# Helper functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def authenticate(username, password):
    user = session.query(User).filter_by(username=username).first()
    if user and verify_password(password, user.password):
        return user
    return None

def create_user(username, password):
    hashed_password = hash_password(password)
    new_user = User(username=username, password=hashed_password)
    session.add(new_user)
    session.commit()

# Streamlit app
st.title("Advanced Personal Monthly Home Financial Review")

# User authentication
st.sidebar.title("User Authentication")
auth_mode = st.sidebar.selectbox("Choose Mode", ["Login", "Sign Up"])

if auth_mode == "Sign Up":
    st.sidebar.subheader("Create a New Account")
    new_username = st.sidebar.text_input("Username")
    new_password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Sign Up"):
        if new_username and new_password:
            create_user(new_username, new_password)
            st.sidebar.success("Account created successfully! Please log in.")
        else:
            st.sidebar.error("Please enter both username and password.")

if auth_mode == "Login":
    st.sidebar.subheader("Log In")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Log In"):
        user = authenticate(username, password)
        if user:
            st.session_state['user_id'] = user.id
            st.sidebar.success(f"Welcome, {username}!")
        else:
            st.sidebar.error("Invalid username or password.")

if 'user_id' in st.session_state:
    # Input Section
    st.header("Add Daily Transaction")
    with st.form(key='transaction_form'):
        date = st.date_input("Date", datetime.date.today())
        category = st.selectbox("Category", ["Income", "Rent", "Utilities", "Groceries", "Transportation", "Entertainment", "Others"])
        amount = st.number_input("Amount", min_value=0.0, value=0.0, step=0.1)
        description = st.text_input("Description")
        submit_button = st.form_submit_button(label='Add Transaction')

        if submit_button:
            new_transaction = Transaction(
                user_id=st.session_state['user_id'],
                date=date,
                category=category,
                amount=amount,
                description=description
            )
            session.add(new_transaction)
            session.commit()
            st.success("Transaction added successfully!")

    # Data Export/Import
    st.header("Export/Import Data")

    # Export data
    if st.button("Export Data"):
        query = session.query(Transaction).filter(Transaction.user_id == st.session_state['user_id'])
        df = pd.read_sql(query.statement, query.session.bind)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='financial_data.csv',
            mime='text/csv',
        )

    # Import data
    uploaded_file = st.file_uploader("Choose a CSV file to import", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        for index, row in data.iterrows():
            new_transaction = Transaction(
                user_id=st.session_state['user_id'],
                date=row['date'],
                category=row['category'],
                amount=row['amount'],
                description=row['description']
            )
            session.add(new_transaction)
        session.commit()
        st.success("Data imported successfully!")

    # Retrieve and display data
    st.header("Monthly Financial Summary")
    selected_month = st.date_input("Select Month", datetime.date.today())

    query = session.query(Transaction).filter(
        Transaction.user_id == st.session_state['user_id'],
        Transaction.date.between(
            datetime.date(selected_month.year, selected_month.month, 1),
            datetime.date(selected_month.year, selected_month.month, 31)
        )
    )

    df = pd.read_sql(query.statement, query.session.bind)

    if not df.empty:
        st.write("### Transactions", df)

        # Monthly Summary
        income = df[df['category'] == 'Income']['amount'].sum()
        total_expenses = df[df['category'] != 'Income']['amount'].sum()
        savings = income - total_expenses

        st.write(f"**Total Income:** ${income:.2f}")
        st.write(f"**Total Expenses:** ${total_expenses:.2f}")
        st.write(f"**Savings:** ${savings:.2f}")

        # Expense Distribution
        st.header("Expense Distribution")
        expense_df = df[df['category'] != 'Income'].groupby('category')['amount'].sum().reset_index()
        fig = px.pie(expense_df, names='category', values='amount', title='Expense Distribution')
        st.plotly_chart(fig)

        # Daily Expense Trend
        st.header("Daily Expense Trend")
        daily_expense_df = df[df['category'] != 'Income'].groupby('date')['amount'].sum().reset_index()
        fig = px.line(daily_expense_df, x='date', y='amount', title='Daily Expenses Over Time')
        st.plotly_chart(fig)

        # Category-wise Bar Chart
        st.header("Category-wise Expense Distribution")
        fig = px.bar(expense_df, x='category', y='amount', title='Category-wise Expense Distribution')
        st.plotly_chart(fig)

        # Budget Goals and Alerts
        st.header("Set Budget Goals")
        with st.form(key='budget_form'):
            budget_category = st.selectbox("Category", ["Rent", "Utilities", "Groceries", "Transportation", "Entertainment", "Others"])
            goal_amount = st.number_input("Goal Amount", min_value=0.0, value=0.0, step=0.1)
            submit_budget_button = st.form_submit_button(label='Set Budget Goal')

            if submit_budget_button:
                new_budget_goal = BudgetGoal(
                    user_id=st.session_state['user_id'],
                    category=budget_category,
                    goal_amount=goal_amount
                )
                session.add(new_budget_goal)
                session.commit()
                st.success("Budget goal set successfully!")

        # Check budget goals
        budget_goals_query = session.query(BudgetGoal).filter(BudgetGoal.user_id == st.session_state['user_id'])
        budget_goals_df = pd.read_sql(budget_goals_query.statement, budget_goals_query.session.bind)

        if not budget_goals_df.empty:
            st.write("### Budget Goals", budget_goals_df)
            for _, row in budget_goals_df.iterrows():
                spent_amount = df[df['category'] == row['category']]['amount'].sum()
                if spent_amount > row['goal_amount']:
                    st.warning(f"You have exceeded your budget for {row['category']} by ${spent_amount - row['goal_amount']:.2f}")
                else:
                    st.info(f"You are within your budget for {row['category']}. Remaining: ${row['goal_amount'] - spent_amount:.2f}")

        # Machine Learning: Forecast Future Expenses
        st.header("Expense Forecasting")
        if len(daily_expense_df) > 1:
            daily_expense_df['days'] = (daily_expense_df['date'] - daily_expense_df['date'].min()).dt.days
            X = daily_expense_df[['days']]
            y = daily_expense_df['amount']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            future_days = np.arange(daily_expense_df['days'].max() + 1, daily_expense_df['days'].max() + 31).reshape(-1, 1)
            future_expenses = model.predict(future_days)

            future_dates = [daily_expense_df['date'].max() + datetime.timedelta(days=int(day)) for day in future_days]
            future_df = pd.DataFrame({'date': future_dates, 'amount': future_expenses})

            fig = px.line(future_df, x='date', y='amount', title='Forecasted Daily Expenses')
            st.plotly_chart(fig)
        else:
            st.write("Not enough data for forecasting. Please add more transactions.")
    else:
        st.write("No transactions found for the selected month.")
