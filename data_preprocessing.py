import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('SuddenChangeInBehaviourAlerts.csv')

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}

df['f.acct_risk'] = df['f.acct_risk'].replace(label_mapping)

df.to_csv('SuddenChangeInBehaviourAlerts_output.csv', index=False)

df = pd.read_csv('SuddenChangeInBehaviourAlerts_output.csv')

train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1

train_df, test_val_df = train_test_split(df, test_size=1 - train_ratio, random_state=42)
test_df, val_df = train_test_split(test_val_df, test_size=val_ratio/(test_ratio + val_ratio), random_state=42)

print("Training sets size: ", train_df.shape[0])
print("Testing sets size: ", test_df.shape[0])
print("Verifying sets size: ", val_df.shape[0])

train_df.to_csv('Data/train.csv', index=False)
test_df.to_csv('Data/test.csv', index=False)
val_df.to_csv('Data/validation.csv', index=False)
