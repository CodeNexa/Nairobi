import matplotlib.pyplot as plt

def visualize_feature_importance(model, feature_names):
    """Plot feature importance for the model."""
    importance = model.feature_importances_
    plt.barh(feature_names, importance)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.show()
