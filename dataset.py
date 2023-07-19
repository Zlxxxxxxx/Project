from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

class FinCrimesDataset(Dataset):
    def __init__(self, filepath, transform=None):
        df = pd.read_csv(filepath)

        feature_columns = ['f.monthly_avg_atm_trans_ct', 'f.daily_atm_tran_ct', 'f.monthly_avg_atm_trans_amt', 'f.daily_atm_tran_amt']
        label_columns = ['f.acct_risk']

        self.features = df[feature_columns]
        self.label = df[label_columns]
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features.iloc[idx, :].values
        label = self.label.iloc[idx].values
        
        if self.transform:
            features, label = self.transform(features), self.transform(label)
            
        return features, label


if __name__ == '__main__':
    filepath = 'Data/train.csv'
    dataset = FinCrimesDataset(filepath)
    dataloader = DataLoader(dataset, batch_size=64)
    print(dataset[0][0].shape)
    print(len(dataloader))