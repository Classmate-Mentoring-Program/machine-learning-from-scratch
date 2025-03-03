import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Génération de données synthétiques (exemple simple)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Variable explicative
y = 4 + 3 * X + np.random.randn(100, 1)  # Relation linéaire avec bruit

# Division des données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions sur les données de test
y_pred = model.predict(X_test)

# Affichage des coefficients
print(f"Coefficient (θ1) : {model.coef_[0][0]}")
print(f"Biais (θ0) : {model.intercept_[0]}")

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Erreur quadratique moyenne (MSE) : {mse}")
print(f"Coefficient de détermination (R²) : {r2}")

# Visualisation des résultats
plt.scatter(X_test, y_test, color="blue", label="Données réelles")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Prédiction")
plt.xlabel("X (Variable explicative)")
plt.ylabel("y (Variable cible)")
plt.legend()
plt.title("Régression Linéaire avec Scikit-learn")
plt.show()