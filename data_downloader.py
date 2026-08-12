"""
Automated Data Downloader for Historical Datasets
Downloads NASDAQ, Kaggle, and other free datasets
"""

import requests
import zipfile
from pathlib import Path
import subprocess

class DatasetDownloader:
    """Download and setup historical datasets."""
    
    def __init__(self):
        self.cache_dir = Path("/Users/aavibharucha/Documents/market_ai/data_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def download_all(self):
        """Download all available datasets."""
        print(" Downloading historical datasets...")
        
        # NASDAQ dataset
        self.download_nasdaq_dataset()
        
        print(" Dataset download complete!")
    
    def download_nasdaq_dataset(self):
        """Download NASDAQ historical dataset from GitHub."""
        try:
            print("Downloading NASDAQ dataset...")
            nasdaq_dir = self.cache_dir / "nasdaq_historical"
            nasdaq_dir.mkdir(exist_ok=True)
            
            # GitHub raw URLs for common tickers
            base_url = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data"
            
            # Download sample tickers (add more as needed)
            sample_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']
            
            for ticker in sample_tickers:
                try:
                    url = f"{base_url}/{ticker}.csv"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        filepath = nasdaq_dir / f"{ticker}.csv"
                        filepath.write_bytes(response.content)
                        print(f"   {ticker}.csv")
                except Exception:
                    continue
            
            print(" NASDAQ dataset ready")
            
        except Exception as e:
            print(f"NASDAQ download error: {e}")
    
    def setup_instructions(self):
        """Print manual setup instructions."""
        return """
         **Manual Dataset Setup Instructions**
        
        For best results, download these datasets manually:
        
        1. **NASDAQ Historical Dataset** (GitHub)
           - Visit: https://github.com/rreichel3/US-Stock-Symbols
           - Download all CSV files
           - Place in: {cache_dir}/nasdaq_historical/
        
        2. **Kaggle Stock Market Dataset**
           - Visit: https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
           - Download dataset
           - Extract to: {cache_dir}/kaggle_stocks/
        
        3. **EODHD API** (Already configured)
           - API Key: 69aa7bd48c84f0.31583834
           - Automatically used for all tickers
        
        The system will automatically fallback between sources.
        """.format(cache_dir=self.cache_dir)

def download_datasets():
    """Main function to download datasets."""
    downloader = DatasetDownloader()
    print(downloader.setup_instructions())
    downloader.download_all()

if __name__ == "__main__":
    download_datasets()
