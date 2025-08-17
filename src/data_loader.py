# File: src/data_loader.py
"""
Advanced data pipeline for M4 dataset with:
- Automated data validation
- Dynamic feature engineering
- Multiple frequency handling
- Efficient data processing
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

class M4Dataset(Dataset):
    """Advanced dataset with temporal features and scaling"""
    
    def __init__(self, series, horizon, freq):
        self.series = series
        self.horizon = horizon
        self.freq = freq
        self.scalers = {}
        self._preprocess()
        
    def _preprocess(self):
        """Advanced preprocessing pipeline"""
        self.processed = []
        self.ids = []
        
        for series_id, ts in self.series.items():
            # Data validation
            if len(ts) < 2 * self.horizon:
                continue
                
            # Create temporal features
            df = pd.DataFrame({
                'value': ts,
                'time': np.arange(len(ts)),
                'series_id': series_id
            })
            
            # Frequency-based features
            if self.freq == 'Hourly':
                df['hour'] = df.index % 24
                df['day_part'] = (df['hour'] // 6).astype(int)
            elif self.freq == 'Daily':
                df['day_of_week'] = df.index % 7
            elif self.freq == 'Monthly':
                df['month'] = df.index % 12
                
            # Rolling statistics
            df['rolling_mean_7'] = df['value'].rolling(window=7, min_periods=1).mean()
            df['rolling_std_7'] = df['value'].rolling(window=7, min_periods=1).std()
            
            # Handle missing values
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            
            # Normalize per-series
            scaler = RobustScaler()
            df['scaled_value'] = scaler.fit_transform(df[['value']])
            self.scalers[series_id] = scaler
            
            # Store processed data
            self.processed.append(df)
            self.ids.append(series_id)
            
    def __len__(self):
        return len(self.processed)
    
    def __getitem__(self, idx):
        df = self.processed[idx]
        series_id = self.ids[idx]
        
        # Prepare model inputs
        encoder_length = len(df) - self.horizon
        decoder_length = self.horizon
        
        return {
            'encoder_cont': df[['scaled_value', 'time']].values[:encoder_length].astype(np.float32),
            'decoder_cont': df[['scaled_value', 'time']].values[encoder_length:].astype(np.float32),
            'encoder_cat': df[['series_id']].values[:encoder_length].astype(int),
            'decoder_cat': df[['series_id']].values[encoder_length:].astype(int),
            'series_id': series_id
        }

class M4DataModule(pl.LightningDataModule):
    """Professional data pipeline for M4 competition data"""
    
    def __init__(self, dataset_name='Hourly', forecast_horizon=24, batch_size=64):
        super().__init__()
        self.dataset_name = dataset_name
        self.horizon = forecast_horizon
        self.batch_size = batch_size
        self.data_dir = "../data"
        
    def prepare_data(self):
        """Automated data download and validation"""
        os.makedirs(self.data_dir, exist_ok=True)
        base_url = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset"
        
        # Download if not exists
        train_file = f"{self.dataset_name}-train.csv"
        test_file = f"{self.dataset_name}-test.csv"
        info_file = "M4-info.csv"
        
        self._download_file(f"{base_url}/Train/{train_file}", train_file)
        self._download_file(f"{base_url}/Test/{test_file}", test_file)
        self._download_file(f"{base_url}/{info_file}", info_file)
        
        # Load data
        self.train_df = pd.read_csv(f"{self.data_dir}/{train_file}")
        self.test_df = pd.read_csv(f"{self.data_dir}/{test_file}")
        self.info_df = pd.read_csv(f"{self.data_dir}/{info_file}")
        
        # Validate dataset
        self._validate_dataset()
        
    def _download_file(self, url, filename):
        """Professional download with validation"""
        path = f"{self.data_dir}/{filename}"
        if not os.path.exists(path):
            print(f"📥 Downloading {filename}...")
            try:
                import requests
                r = requests.get(url)
                with open(path, 'wb') as f:
                    f.write(r.content)
                print(f"✅ Downloaded {filename} successfully")
            except Exception as e:
                print(f"❌ Download failed: {e}")
                raise
                
    def _validate_dataset(self):
        """Data quality checks"""
        assert len(self.train_df) > 0, "Train dataset is empty"
        assert len(self.test_df) > 0, "Test dataset is empty"
        assert 'V1' in self.train_df.columns, "Invalid data format"
        print(f"✅ Data validated: {len(self.train_df)} series found")
        
    def setup(self, stage=None):
        """Feature engineering and dataset creation"""
        # Convert to time series dictionary
        self.series = {}
        for i, row in self.train_df.iterrows():
            series_id = row['V1']
            values = row.drop('V1').dropna().values
            self.series[series_id] = values
            
        # Create dataset
        self.dataset = M4Dataset(
            series=self.series,
            horizon=self.horizon,
            freq=self.dataset_name
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self):
        # Use same data for validation in this implementation
        return self.train_dataloader()
