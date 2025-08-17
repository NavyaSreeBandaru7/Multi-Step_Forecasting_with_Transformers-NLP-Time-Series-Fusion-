# File: src/main.py
"""
M4 Forecasting with Temporal Fusion Transformers
- Advanced multi-horizon forecasting
- NLP integration for explainable forecasts
- Generative AI components
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

from data_loader import M4DataModule
from tft_model import TemporalFusionTransformer
from lightning_trainer import PLTrainer
from forecast_explainer import ForecastExplainer

def main():
    parser = argparse.ArgumentParser(description='TFT Forecasting Pipeline')
    parser.add_argument('--dataset', type=str, default='Hourly', help='M4 dataset frequency')
    parser.add_argument('--horizon', type=int, default=24, help='Forecasting horizon')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"🚀 Starting TFT Forecasting: {args.dataset} dataset")
    print(f"📈 Forecast Horizon: {args.horizon} steps")
    print(f"⚙️ Parameters: {args.epochs} epochs, batch size {args.batch_size}")
    print(f"{'='*50}\n")

    # Data preparation
    print("🔍 Loading and preprocessing data...")
    dm = M4DataModule(
        dataset_name=args.dataset,
        forecast_horizon=args.horizon,
        batch_size=args.batch_size
    )
    dm.prepare_data()
    dm.setup()
    
    # Model initialization
    print("\n🧠 Initializing Temporal Fusion Transformer...")
    model = TemporalFusionTransformer(
        dataset=dm.dataset,
        hidden_size=64,
        lstm_layers=2,
        attention_head_size=4,
        dropout=0.1
    )
    
    # Training
    print("\n⚡ Starting training...")
    trainer = PLTrainer(
        model=model,
        datamodule=dm,
        max_epochs=args.epochs,
        gpus=args.gpus
    )
    trainer.train()
    
    # Forecasting
    print("\n🔮 Generating forecasts...")
    forecasts = trainer.predict()
    
    # Explainability
    print("\n💡 Generating explainable insights...")
    explainer = ForecastExplainer(model, dm)
    explanations = explainer.explain(forecasts)
    
    # NLP Report
    print("\n📝 Creating NLP report...")
    report = explainer.generate_nlp_report(explanations)
    
    print("\n✅ Pipeline completed successfully!")
    print(f"📊 Forecast results saved to: results/{args.dataset}_forecasts.csv")
    print(f"📝 NLP report saved to: reports/{args.dataset}_forecast_report.txt")

if __name__ == "__main__":
    main()
