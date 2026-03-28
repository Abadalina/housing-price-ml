"""
Model evaluation utilities.
Author: Alejandro Abadal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PALETTE = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
PRIMARY  = "#2a9d8f"
ACCENT   = "#e76f51"


def regression_metrics(y_true, y_pred, label: str = "") -> dict:
    """Compute MAE, RMSE, MAPE and R2."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100

    metrics = {"Model": label, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R2": r2}
    return metrics


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Build a comparison dataframe from a list of metric dicts."""
    df = pd.DataFrame(results).set_index("Model")
    return df.sort_values("R2", ascending=False).round(3)


def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Bar chart comparing model metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["MAE", "RMSE", "R2"]
    colors  = [ACCENT, ACCENT, PRIMARY]

    for ax, metric, color in zip(axes, metrics, colors):
        data = results_df[metric].sort_values(ascending=(metric != "R2"))
        bars = ax.barh(data.index, data.values, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_title(metric)
        ax.set_xlabel(metric)

    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_predictions(y_true, y_pred, model_name: str, save_path: str = None):
    """Actual vs predicted scatter plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color=PRIMARY)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, 'r--', lw=1.5, label='Perfect prediction')
    axes[0].set_xlabel('Actual price (€)')
    axes[0].set_ylabel('Predicted price (€)')
    axes[0].set_title(f'{model_name} — Actual vs Predicted')
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}€'))
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}€'))

    # Residuals
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=50, color=ACCENT, alpha=0.8, edgecolor='white')
    axes[1].axvline(0, color='black', lw=1.5, ls='--')
    axes[1].set_xlabel('Residual (€)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'{model_name} — Residuals distribution')
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}€'))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 20,
                            model_name: str = "", save_path: str = None):
    """Horizontal bar chart of feature importances."""
    if not hasattr(model, 'feature_importances_'):
        print(f"  {model_name} does not expose feature_importances_")
        return

    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    bars = ax.barh(importance.index, importance.values, color=PRIMARY, alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel('Feature importance')
    ax.set_title(f'{model_name} — Top {top_n} feature importances')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
