from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")