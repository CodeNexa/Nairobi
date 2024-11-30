def calculate_integrity_score(honesty, transparency, accountability, ethics, consistency):
    """Calculate a weighted composite integrity score."""
    weights = {"honesty": 0.2, "transparency": 0.2, "accountability": 0.3, "ethics": 0.2, "consistency": 0.1}
    score = (honesty * weights["honesty"] +
             transparency * weights["transparency"] +
             accountability * weights["accountability"] +
             ethics * weights["ethics"] +
             consistency * weights["consistency"])
    return round(score, 2)
