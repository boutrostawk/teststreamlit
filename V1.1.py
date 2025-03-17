import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def fetch_stock_data(ticker, start_date, end_date):
    # Download historical data from yfinance
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    df.reset_index(inplace=True)  # Ensure date is a column
    return df[['Date', 'Open', 'Close']]

def save_to_excel(data_dict, output_file):
    # Save each ticker's data to a separate Excel sheet
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for ticker, df in data_dict.items():
            df.to_excel(writer, sheet_name=ticker) # Removed index=False

def plot_stock_with_sma(df, ticker, sma_windows=[10, 20]):
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the closing price
    ax.plot(df['Date'], df['Close'], label='Close Price')
    
    # Calculate and plot SMAs for each specified window
    for window in sma_windows:
        sma_label = f'SMA_{window}'
        df[sma_label] = df['Close'].rolling(window=window).mean()
        ax.plot(df['Date'], df[sma_label], label=sma_label)
    
    # Set plot title and labels
    ax.set_title(f"{ticker} Price and SMA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    
    # Return the figure for display in Streamlit
    return fig
    
    # Calculate and plot each SMA
    for window in sma_windows:
        sma_label = f'SMA_{window}'
        df[sma_label] = df['Close'].rolling(window=window).mean()
        plt.plot(df['Date'], df[sma_label], label=sma_label)
    
    plt.title(f"{ticker} Price and SMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage:
tickers = ['AAPL', 'MSFT', 'GOOGL']
start_date = '2023-01-01'
end_date = '2023-03-01'

stock_data = {}
for ticker in tickers:
    df = fetch_stock_data(ticker, start_date, end_date)
    stock_data[ticker] = df
    plot_stock_with_sma(df, ticker)

save_to_excel(stock_data, "stock_data.xlsx")

def generate_signal(df, short_window=10, long_window=20):
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    
    # Simple signal based on last available values
    if df['SMA_short'].iloc[-1] > df['SMA_long'].iloc[-1]:
        return "Buy"
    elif df['SMA_short'].iloc[-1] < df['SMA_long'].iloc[-1]:
        return "Sell"
    else:
        return "Hold"

# Example for one ticker:
for ticker, df in stock_data.items():
    signal = generate_signal(df)
    print(f"{ticker}: {signal}")

    import streamlit as st

# --- Password Protection ---
# (For demonstration, we use a hardcoded password. In production, consider using environment variables or a package like streamlit-authenticator.)
PASSWORD = "A&B-Rev1.1"

def check_password():
    if "password_entered" not in st.session_state:
        st.session_state["password_entered"] = False
        
    if not st.session_state["password_entered"]:
        pwd = st.text_input("Enter password", type="password")
        if st.button("Submit"):
            if pwd == PASSWORD:
                st.session_state["password_entered"] = True
            else:
                st.error("Incorrect password")
        return False
    return True

# --- Main App ---
if check_password():
    st.title("Project Z by A&B")

    # Inputs
    tickers = st.text_input("Enter tickers separated by commas", "AAPL, MSFT, GOOGL").split(",")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-03-01"))
    
    if st.button("Run Analysis"):
        stock_data = {}
        for ticker in [t.strip().upper() for t in tickers]:
            df = fetch_stock_data(ticker, start_date, end_date)
            stock_data[ticker] = df
        
            st.subheader(f"{ticker} Price and SMA")
            # Create the figure with the plot
            fig = plot_stock_with_sma(df, ticker)
            # Display the figure using Streamlit
            st.pyplot(fig)
        
            signal = generate_signal(df)
            st.write(f"Trading Signal for {ticker}: **{signal}**")
    
        # Optionally, let user download the Excel file:
        save_to_excel(stock_data, "stock_data.xlsx")
        with open("stock_data.xlsx", "rb") as file:
            st.download_button("Download Excel File", file, file_name="stock_data.xlsx")
