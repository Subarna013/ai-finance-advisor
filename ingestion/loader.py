from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split():
    urls = [
        # Core finance concepts
        "https://www.investopedia.com/terms/s/stockmarket.asp",
        "https://www.investopedia.com/terms/i/inflation.asp",
        "https://www.investopedia.com/terms/r/risk.asp",
        "https://www.investopedia.com/terms/p/portfolio.asp",
        "https://www.investopedia.com/terms/d/diversification.asp",

        # Indian context (VERY IMPORTANT 🔥)
        "https://www.rbi.org.in/Scripts/FAQView.aspx?Id=28",
        "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0",

        # Personal finance
        "https://www.investopedia.com/personal-finance-4427760",
        "https://www.investopedia.com/terms/m/mutualfund.asp",
        "https://www.investopedia.com/terms/e/equity.asp",
    ]

    # Load documents
    docs = [WebBaseLoader(url).load() for url in urls]
    doc_list = [item for sublist in docs for item in sublist]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50  # small overlap improves quality 🔥
    )

    return splitter.split_documents(doc_list)