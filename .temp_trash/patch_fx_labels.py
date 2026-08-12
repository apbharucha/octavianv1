src = open('financial_llm_engine.py').read()

old = """    _PAIR_LABEL = {
        \"EURUSD=X\": \"Euro / US Dollar (EURUSD)\", \"USDJPY=X\": \"US Dollar / Yen (USDJPY)\",
        \"GBPUSD=X\": \"British Pound / US Dollar (GBPUSD)\", \"AUDUSD=X\": \"Aussie / US Dollar (AUDUSD)\",
        \"USDCAD=X\": \"US Dollar / Canadian Dollar (USDCAD)\", \"USDCHF=X\": \"US Dollar / Swiss Franc (USDCHF)\",
        \"NZDUSD=X\": \"Kiwi / US Dollar (NZDUSD)\", \"EURGBP=X\": \"Euro / Pound (EURGBP)\",
        \"EURJPY=X\": \"Euro / Yen (EURJPY)\", \"GBPJPY=X\": \"Pound / Yen (GBPJPY)\",
    }"""
new = """    _PAIR_LABEL = {
        \"EURUSD=X\": \"Euro / US Dollar (EURUSD)\", \"USDJPY=X\": \"US Dollar / Yen (USDJPY)\",
        \"GBPUSD=X\": \"British Pound / US Dollar (GBPUSD)\", \"AUDUSD=X\": \"Aussie / US Dollar (AUDUSD)\",
        \"USDCAD=X\": \"US Dollar / Canadian Dollar (USDCAD)\", \"USDCHF=X\": \"US Dollar / Swiss Franc (USDCHF)\",
        \"NZDUSD=X\": \"Kiwi / US Dollar (NZDUSD)\", \"EURGBP=X\": \"Euro / Pound (EURGBP)\",
        \"EURJPY=X\": \"Euro / Yen (EURJPY)\", \"GBPJPY=X\": \"Pound / Yen (GBPJPY)\",
        \"USDBRL=X\": \"US Dollar / Brazilian Real (USDBRL)\", \"USDMXN=X\": \"US Dollar / Mexican Peso (USDMXN)\",
        \"USDTRY=X\": \"US Dollar / Turkish Lira (USDTRY)\", \"USDZAR=X\": \"US Dollar / South African Rand (USDZAR)\",
        \"USDINR=X\": \"US Dollar / Indian Rupee (USDINR)\", \"USDCNY=X\": \"US Dollar / Chinese Yuan (USDCNY)\",
        \"USDKRW=X\": \"US Dollar / Korean Won (USDKRW)\", \"USDSGD=X\": \"US Dollar / Singapore Dollar (USDSGD)\",
        \"USDHKD=X\": \"US Dollar / Hong Kong Dollar (USDHKD)\", \"EURBRL=X\": \"Euro / Brazilian Real (EURBRL)\",
    }"""
assert src.count(old) == 1
src = src.replace(old, new)
open('financial_llm_engine.py', 'w').write(src)
print("PATCH_OK")
