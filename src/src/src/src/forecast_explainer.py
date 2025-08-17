# File: src/forecast_explainer.py
"""
Advanced forecast interpretation with:
- SHAP values for feature importance
- Attention visualization
- Generative AI explanations
- NLP report generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import shap
import torch

class ForecastExplainer:
    """Comprehensive forecast interpretation toolkit"""
    
    def __init__(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule
        self.explainer = None
        self.nlp_generator = pipeline(
            "text-generation", 
            model="gpt2-medium",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def explain(self, forecasts):
        """Generate multiple explanations"""
        # Feature importance with SHAP
        self._shap_analysis()
        
        # Attention visualization
        self._attention_analysis()
        
        # Return interpretation results
        return {
            "shap_values": self.shap_values,
            "attention_weights": self.attention_weights
        }
    
    def _shap_analysis(self):
        """Compute SHAP values for feature importance"""
        # Sample data for explanation
        sample = next(iter(self.datamodule.val_dataloader()))
        background = sample['encoder_cont'][:10]
        
        # Create explainer
        self.explainer = shap.DeepExplainer(
            self.model,
            background.to(self.model.device)
        
        # Compute SHAP values
        self.shap_values = self.explainer.shap_values(
            sample['encoder_cont'][:2].to(self.model.device))
        
        # Visualize
        self._plot_shap_summary()
        
        return self.shap_values
    
    def _attention_analysis(self):
        """Extract and visualize attention weights"""
        # Hook to capture attention weights
        attention_weights = []
        
        def hook(module, input, output):
            _, weights = output
            attention_weights.append(weights.detach().cpu())
        
        hook_handle = self.model.multihead_attn.register_forward_hook(hook)
        
        # Forward pass
        sample = next(iter(self.datamodule.val_dataloader()))
        self.model(sample)
        
        # Remove hook
        hook_handle.remove()
        
        # Process weights
        self.attention_weights = attention_weights[0]
        self._plot_attention_heatmap()
        
        return self.attention_weights
    
    def generate_nlp_report(self, explanations):
        """Generate natural language forecast report"""
        # Get sample forecast
        sample = next(iter(self.datamodule.val_dataloader()))
        forecast = self.model.predict(sample)[0, :, 1].cpu().numpy()  # Median forecast
        
        # Generate text explanation
        prompt = f"""
        As a senior forecasting analyst, explain the following time series forecast:
        
        Dataset: {self.datamodule.dataset_name}
        Forecast horizon: {self.datamodule.horizon} steps
        Key features:
        - Trend: {'upward' if np.mean(np.diff(forecast)) > 0 else 'downward'}
        - Volatility: {np.std(forecast):.2f}
        - Key patterns: {self._identify_patterns(forecast)}
        
        Provide a professional analysis for business stakeholders:
        """
        
        report = self.nlp_generator(
            prompt,
            max_length=300,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9
        )[0]['generated_text']
        
        # Save report
        os.makedirs("../reports", exist_ok=True)
        with open(f"../reports/{self.datamodule.dataset_name}_forecast_report.txt", "w") as f:
            f.write(report)
            
        return report
    
    def _plot_shap_summary(self):
        """Create professional SHAP summary plot"""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, 
            self.sample['encoder_cont'][:2].cpu().numpy(),
            feature_names=['Value', 'Time'],
            show=False
        )
        plt.title('Feature Importance for TFT Forecasting')
        plt.tight_layout()
        plt.savefig(f"../results/{self.datamodule.dataset_name}_shap_summary.png")
        plt.close()
    
    def _plot_attention_heatmap(self):
        """Visualize attention patterns"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            self.attention_weights[0].mean(dim=0).numpy(),
            cmap='viridis',
            annot=True,
            fmt=".2f"
        )
        plt.title('Attention Weights Across Time Steps')
        plt.xlabel('Encoder Time Steps')
        plt.ylabel('Decoder Time Steps')
        plt.tight_layout()
        plt.savefig(f"../results/{self.datamodule.dataset_name}_attention_heatmap.png")
        plt.close()
    
    def _identify_patterns(self, forecast):
        """Detect common time series patterns"""
        # Simple pattern detection
        diff = np.diff(forecast)
        if np.all(diff > 0):
            return "Strong upward trend"
        elif np.all(diff < 0):
            return "Strong downward trend"
        elif np.mean(diff) > 0 and np.std(diff) > np.mean(forecast)*0.1:
            return "Volatile with upward bias"
        else:
            return "No clear trend with fluctuations"
