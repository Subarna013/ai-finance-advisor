import yfinance as yf

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "name": info.get("longName"),
            "price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "sector": info.get("sector")
        }

    except Exception as e:
        return {"error": str(e)}