import sys
sys.path.insert(0, "/Users/aavibharucha/Documents/octavian")
from futures_engine import get_futures_engine

eng = get_futures_engine()
eng.rf_rate = 0.05

ok = True
violations = 0
checked = 0
worst_eep = 0.0  # max early-exercise premium observed

for typ in ["call", "put"]:
    for F in [50, 80, 95, 100, 105, 120, 150]:
        for K in [100]:
            for T in [0.05, 0.25, 1.0, 2.0]:
                for sig in [0.1, 0.2, 0.35]:
                    checked += 1
                    euro = eng.black76(F, K, T, sig, typ)["price"]
                    am = eng.whaley_american_futures(F, K, T, sig, typ)
                    intrinsic = max(0.0, F - K) if typ == "call" else max(0.0, K - F)
                    if am < euro - 1e-9:
                        violations += 1
                        print(f"VIOLATION am<euro {typ} F={F} K={K} T={T} s={sig}: euro={euro:.4f} am={am:.4f}")
                        ok = False
                    if am < intrinsic - 1e-9:
                        violations += 1
                        print(f"VIOLATION am<intrinsic {typ} F={F} K={K} T={T} s={sig}: int={intrinsic:.4f} am={am:.4f}")
                        ok = False
                    worst_eep = max(worst_eep, am - euro)

print(f"checked={checked} violations={violations}")
print(f"max early-exercise premium observed: {worst_eep:.4f}")
print("PASS" if ok else "FAIL")
