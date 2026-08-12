import yfinance as yf
df = yf.download('OIL', period='3mo')
print(f"OIL shape: {df.shape}")
df = yf.download('CL=F', period='3mo')
print(f"CL=F shape: {df.shape}")
