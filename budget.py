import google.generativeai as genai
import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============ Gemini Setup ============
genai.configure(api_key="AIzaSyDg-nYA54jfBUywmGoHR8Kvsirt5Z6GhEk")  # Replace with env var if deploying
llm_model = genai.GenerativeModel("gemini-2.5-flash")

# ============ Streamlit UI ============
st.title("💸 Budget Buddy: Your Smart Guide to Budgeting")

st.write("Answer the questions below to get a personalized financial plan and local resources from your AI financial advisor.")

# Collect user input
age = st.text_input("How old are you?")
income = st.text_input("Enter your monthly income (USD):")
expenses = st.text_input("Enter your estimated monthly expenses (USD):")
savings = st.text_input("How much do you currently have saved?")
investments = st.text_input("Do you have any investments? If yes, describe them briefly:")
debt = st.text_input("Do you have any debt? If yes, how much and what are the interest rates?")
goal = st.text_input("What is your main financial goal? (e.g., saving for college, paying debt)")
risk = st.selectbox("What is your risk tolerance?", ["Low", "Medium", "High"])
location = st.text_input("Enter your city or region:")
upcoming = st.text_input("Do you have any major upcoming expenses? (e.g., tuition, car, moving)")

# Button for budget advice
if st.button("Get Budget Advice"):
    inputs = [age, income, expenses, savings, investments, debt, goal, risk, location, upcoming]
    if not all(field.strip() for field in inputs):
        st.warning("Please fill out all fields.")
    else:
        prompt = f"""
        Act as a certified financial advisor. Use the information below to give a 
        tailored, professional budgeting analysis. 

        User Details:
        - Age: {age}
        - Monthly Income: ${income}
        - Monthly Expenses: ${expenses}
        - Current Savings: ${savings}
        - Current Investments: {investments}
        - Debt: {debt}
        - Financial Goal: {goal}
        - Risk Tolerance: {risk}
        - Location: {location}
        - Major Upcoming Expenses: {upcoming}

        Organize your response into these sections:
        1. In-Depth Summary of Current Financial Situation 
        2. Personalized Budgeting Recommendations (e.g., what % to save, spend, invest)
        3. Suggestions for Adjustments (what to change about their income, spending, saving, or investment habits)
        4. Action Plan (3 practical next steps they should take) 
        5. Local or Free Resources (based on location)

        Give clear, concise, and practical advice. If important information is missing, ask for it. 
        """

        response = llm_model.generate_content(prompt)
        st.subheader("📊 Budget Buddy’s Advice:")
        st.write(response.parts[0].text if hasattr(response, "parts") else response.text)

# ============ Sidebar: Stock Investment Analyzer ============
st.sidebar.header("📈 Stock Investment Analyzer")

ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)")

if ticker:
    st.sidebar.write(f"Analyzing: **{ticker.upper()}**")

    try:
        # Fetch data from Yahoo Finance
        data = yf.download(ticker, start="2015-01-01", end=pd.Timestamp.today())
        data = data[['Close']].dropna()
        data['Previous Close'] = data['Close'].shift(1)
        data = data.dropna()

        # ======= Show Stock Graph in Sidebar =======
        st.sidebar.subheader("📉 Price Chart")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(data.index, data['Close'], color='blue')
        ax.set_title(f"{ticker.upper()} Closing Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.tick_params(axis='x', labelsize=6)
        ax.grid(True)
        st.sidebar.pyplot(fig)

        # ======= Prepare data for regression =======
        X = data[['Previous Close']]
        y = data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Linear Regression Model
        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)

        # Predict next day price
        last_close = float(data.iloc[-1]['Close'])
        predicted_next = float(reg_model.predict(pd.DataFrame([[last_close]], columns=["Previous Close"]))[0])

        # Show results
        st.sidebar.subheader("🔮 Prediction")
        st.sidebar.write(f"Last Close: **${last_close:.2f}**")
        st.sidebar.write(f"Predicted Next Day Close: **${predicted_next:.2f}**")

        if predicted_next > last_close:
            st.sidebar.success("✅ Suggestion: Consider investing. Trend shows growth.")
        else:
            st.sidebar.warning("⚠️ Suggestion: Hold off. Price may decrease or stay flat.")

    except Exception as e:
        st.sidebar.error(f"Error: {e}")
