def build_watchlist(sectors, fx, futures):
    watch = []
    watch += sectors[:3]
    watch += fx[:3]
    watch += futures[:2]
    return watch
