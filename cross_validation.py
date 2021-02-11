import config
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

df = pd.read_csv(config.TRAIN_CSV_PATH)

skf = StratifiedKFold(2, shuffle=True, random_state=42)
df['fold'] = -1
for i, (train_idx, valid_idx) in enumerate(skf.split(df, df['label'])):
    df.loc[valid_idx, 'fold'] = i
df.to_csv('folds.csv', index=False)
df.head()