import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import joblib, os, sys

os.makedirs('model', exist_ok=True)

# Use AI run if present, else PoT. Allow override via argv.
src = 'logs/aipot.csv' if os.path.exists('logs/aipot.csv') else 'logs/pot.csv'
if len(sys.argv) == 2:
    src = sys.argv[1]

df = pd.read_csv(src)

# Recreate log_size if missing
if 'log_size' not in df.columns:
    if 'size' not in df.columns:
        raise SystemExit("No size/log_size in input CSV; run a fresh scheduler to generate richer logs.")
    df['log_size'] = np.log(df['size'] + 1.0)

# Interactions to help a linear model approximate nonlinearity
df['op_log']  = df['op'] * df['log_size']
df['seq_log'] = df['seq'] * df['log_size']

# Stable target in log-space
y = np.log(df['actual_ms'].clip(lower=0.1).values)
X = df[['log_size','op','seq','op_log','seq_log']].values

model = Ridge(alpha=0.01)
model.fit(X, y)

joblib.dump(model, 'model/model.joblib')

# Export intercept + 5 coefficients
with open('model/coeffs.txt','w') as f:
    coeffs = [model.intercept_] + list(model.coef_)
    f.write(" ".join(str(c) for c in coeffs) + "\n")

print(f"Trained on {src}. Wrote model/coeffs.txt")