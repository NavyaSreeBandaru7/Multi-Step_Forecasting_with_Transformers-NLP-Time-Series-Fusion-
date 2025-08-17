# File: src/lightning_trainer.py
"""
Professional training pipeline with:
- Mixed precision training
- Gradient clipping
- Learning rate scheduling
- Early stopping
- Model checkpointing
- WandB integration
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PLTrainer(pl.LightningModule):
    """Production-grade training pipeline"""
    
    def __init__(self, model, datamodule, max_epochs=50, gpus=1):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.lr = 1e-3
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # Model forward pass
        predictions = self(batch)
        
        # Target values (last horizon steps)
        targets = batch['decoder_cont'][:, :, 0]  # Scaled value at index 0
        
        # Quantile loss (pinball loss)
        losses = []
        quantiles = [0.1, 0.5, 0.9]
        for i, q in enumerate(quantiles):
            error = targets - predictions[:, :, i]
            loss = torch.max((q-1)*error, q*error).mean()
            losses.append(loss)
            
        total_loss = sum(losses) / len(losses)
        
        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def train(self):
        """Complete training workflow"""
        # Callbacks
        early_stop = EarlyStopping(
            monitor='train_loss',
            patience=7,
            verbose=True,
            mode='min'
        )
        
        checkpoint = ModelCheckpoint(
            dirpath='checkpoints/',
            filename='best_tft_model',
            save_top_k=1,
            monitor='train_loss',
            mode='min'
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Logger
        wandb_logger = WandbLogger(project='tft-forecasting', log_model=True)
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            gpus=self.gpus,
            logger=wandb_logger,
            callbacks=[early_stop, checkpoint, lr_monitor],
            gradient_clip_val=0.1,
            precision=16,
            deterministic=True
        )
        
        # Execute training
        trainer.fit(self, self.datamodule)
        
        # Load best model
        self.model.load_state_dict(torch.load(checkpoint.best_model_path))
        return trainer
    
    def predict(self):
        """Generate forecasts"""
        dataloader = self.datamodule.val_dataloader()
        return self.model.predict(dataloader)
