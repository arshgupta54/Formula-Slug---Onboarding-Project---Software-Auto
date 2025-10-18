
# Valorant FFA Winner Predictor

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/gupta-v/valorant-performance-predictor/refs/heads/main/data/val_stats.csv"
df = pd.read_csv(url)


for col in df.columns:
    if df[col].dtype == 'object' and df[col].astype(str).str.contains(',').any():
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')

rank_order = {'Iron': 1, 'Bronze': 2, 'Silver': 3, 'Gold': 4, 'Platinum': 5, 
              'Diamond': 6, 'Ascendant': 7, 'Immortal': 8, 'Radiant': 9}
if 'rating' in df.columns and df['rating'].dtype == object:
    df['rating_encoded'] = df['rating'].astype(str).str.split().str[0].map(rank_order)
    df = df.drop(columns=['rating'])
elif 'rating' in df.columns and df['rating'].dtype == float:
    df = df.rename(columns={'rating': 'rating_score'})

columns_to_drop = ['region', 'name', 'tag', 'agent_1', 'agent_2', 'agent_3',         
                   'gun1_name', 'gun2_name', 'gun3_name', 'wins', 'win_percent']

df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
df_cleaned = df_cleaned.dropna()


# 4. Features (X) and Target (y) - Target is KILLS PER ROUND
TARGET_COLUMN = 'kills_round'
X = df_cleaned.drop(columns=[TARGET_COLUMN])
y = df_cleaned[TARGET_COLUMN]

# Split data and Standardize features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

class ValorantNN_Regression(nn.Module):
    def __init__(self, input_size):
        super(ValorantNN_Regression, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x

model = ValorantNN_Regression(X_train_scaled.shape[1])
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss (MSE): {loss.item():.6f}')


model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)

# Convert to numpy for metrics and display
y_test_np = y_test_tensor.numpy().flatten()
y_pred_np = y_pred.numpy().flatten()

# Regression Metrics
mse = mean_squared_error(y_test_np, y_pred_np)
rmse = np.sqrt(mse) 
r2 = r2_score(y_test_np, y_pred_np)

print('\n--- Model Evaluation (Regression) ---')
print(f'Test Mean Squared Error (MSE): {mse:.6f}')
print(f'Test Root Mean Squared Error (RMSE): {rmse:.6f}')
print(f'Test R-squared (R² Score): {r2:.4f}')

# --- Declare the Predicted FFA Winner ---
# 1. Find the index (row number) of the maximum predicted kills/round score.
winner_index = np.argmax(y_pred_np)
winner_predicted_kpr = y_pred_np[winner_index]
winner_actual_kpr = y_test_np[winner_index]

print("\n----------------------------------------------------")
print(f"PREDICTED FFA WINNER (from Test Set):")
print(f"The row with index [{winner_index}] is the predicted winner.")
print(f"Predicted Kills/Round: {winner_predicted_kpr:.3f}")
print(f"Actual Kills/Round (for this player): {winner_actual_kpr:.3f}")
print("----------------------------------------------------")

# --- Display Sample Predictions ---
print('\n--- Sample Predicted Kills Per Round (FFA Performance) ---')
results = pd.DataFrame({
    'Actual Kills/Round': y_test_np,
    'Predicted Kills/Round': y_pred_np
})

# Display the first 10 predictions
print(results.head(10).style.format({'Actual Kills/Round': '{:.3f}', 'Predicted Kills/Round': '{:.3f}'}))