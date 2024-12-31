import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import json

functions_infos = [
    {
        "type": "function",
          "function": {
            "name": "plot_stock_prices",
            "description": "Fetch stock data for a company and plot its closing prices over a specified period. Call this function to visualize stock performance based on \
                  historical data. the function will plot the data itself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple).",
                    },
                    "period": {
                        "type": "string",
                        "description": "The time period for which to retrieve stock prices. Must be one of the following: ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'].",
                    },
                    "column_to_plot": {
                        "type": "string",
                        "description": "the column of the stock data to plot. One of Open, High, Low, Close, Volume",
                    },
                },
                "required": ["ticker", "period", "column_to_plot"],
                "additionalProperties": False,
            }
        }
    },
    {
        "type": "function",
          "function": {
            "name": "get_sustainability_scores",
            "description": "Retrieves sustainability scores for the company. This function provides insights into the company's environmental, social, and governance (ESG) performance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple).",
                    },
                },
                "required": ["ticker"],
                "additionalProperties": False,
            }
        }
    },
    {
        "type": "function",
          "function":  {
            "name": "get_institutional_holders",
            "description": "Retrieves institutional holder information for the specified company. This function provides details about institutional ownership, including the percentage held and the value of shares.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple).",
                    },
                },
                "required": ["ticker"],
                "additionalProperties": False,
            }
        }
    }
]


def plot_stock_prices(ticker: str, period: str, column_to_plot: str):
    """
    Fetch stock data for a company and plot it.

    Args:
    ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple).
    days (int): Number of past days to retrieve stock prices must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

    Returns:
    Histogram: Displays a plot of the stock prices.
    """
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period=period)
    plt.figure(figsize=(10, 6))
    st.line_chart(hist, y=[column_to_plot])
    return hist

def get_sustainability_scores(ticker: str):
    """
    Retrieves sustainability scores for the company.

    Args:
    ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple).

    Returns:
    Dictionary: a dictionary that contains sustainability scores of the company.
    """
    ticker = yf.Ticker(ticker)
    return ticker.sustainability.to_dict()

def get_institutional_holders(ticker: str):
    """
    Retrieves institutional holder information.

    Args:
    ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple).

    Returns:
    Dictionary: a dictionary that contains institutional holder information (Date Reported, Holder, pctHeld, Shares, Value)
    """
    print("ticker is: ", ticker)
    ticker = yf.Ticker(ticker)
    return ticker.institutional_holders.to_dict()


functions_dict = {
    "plot_stock_prices": plot_stock_prices,
    "get_sustainability_scores": get_sustainability_scores,
    "get_institutional_holders": get_institutional_holders
}

def handle_tool_call(response):
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls and len(tool_calls) > 0 :
        tools_data = []
        for tool_call in tool_calls:
            arguments = json.loads(tool_call.function.arguments) if len(tool_call.function.arguments) > 0 else {}
            func = functions_dict.get(tool_call.function.name)
            tool_response = func(**arguments)
            if not isinstance(tool_response, str):
                tool_response = json.dumps(tool_response, default=str)
            tools_data.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": tool_response
            })
        return tools_data 
    return None  

