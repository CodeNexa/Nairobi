from src.preprocess import load_data, preprocess_data
from src.train import train_integrity_model
from src.predict import predict_integrity
from src.scoring import calculate_integrity_score
from src.visualize import plot_integrity_scores

# Load dataset
df = load_data("data/kenya_rideshare_data.csv")
df = preprocess_data(df)

# Train model
model = train_integrity_model(df)

# Evaluate platforms
platforms = ["Bolt", "Faras", "Uber"]
scores = [
    calculate_integrity_score(80, 70, 60, 75, 85),
    calculate_integrity_score(70, 60, 55, 65, 70),
    calculate_integrity_score(90, 85, 80, 95, 90)
]

# Visualize results
plot_integrity_scores(scores, platforms)
