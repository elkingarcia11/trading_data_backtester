import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class TradingHeatMapAnalyzer:
    def __init__(self, csv_file_path):
        """
        Initialize the analyzer with CSV data
        
        Parameters:
        csv_file_path (str): Path to the CSV file containing trading data
        """
        self.df = pd.read_csv(csv_file_path)
        
        # Define what columns are actually indicators vs target metrics
        self.target_metrics = ['win_rate', 'average_trade_profit', 'total_trades', 'max_drawdown', 'average_trade_duration (minutes)', 'expected_profit', 'total_trade_profit', 'profit_per_trade_ratio', 'win_loss_ratio', 'profit_to_drawdown_ratio', 'trades_per_day', 'profit_per_day', 'profit_per_minute']
        
        # Define indicator columns (parameters that can be varied)
        indicator_candidates = ['ema', 'vwma', 'sma', 'macd_line', 'macd_signal', 'stoch_rsi_k', 'stoch_rsi_d', 'stoch_rsi_period', 'roc', 'roc_of_roc', 'rsi', 'volatility', 'price_change', 'atr', 'macd_fast', 'macd_slow']
        
        # Only include indicator columns that actually exist in the data
        self.indicator_columns = [col for col in indicator_candidates if col in self.df.columns]
        
        # Clean and validate data
        self._prepare_data()
    
    def _prepare_data(self):
        """Clean and prepare the data for analysis"""
        # Check which target metrics actually exist in the data
        available_target_metrics = [col for col in self.target_metrics if col in self.df.columns]
        missing_metrics = [col for col in self.target_metrics if col not in self.df.columns]
        
        if missing_metrics:
            print(f"Warning: Missing target metrics: {missing_metrics}")
            print("These will be calculated if possible, or skipped if not available.")
        
        # Remove any rows with missing values in key columns that actually exist
        # Filter out empty columns from required_cols
        non_empty_cols = []
        for col in self.indicator_columns + available_target_metrics:
            if col in self.df.columns and not self.df[col].isna().all():
                non_empty_cols.append(col)
        
        if non_empty_cols:
            # Only drop rows that have NaN in the non-empty columns
            self.df = self.df.dropna(subset=non_empty_cols)
            print(f"Data after cleaning: {len(self.df)} rows (removed rows with NaN in {len(non_empty_cols)} key columns)")
        else:
            print("Warning: All columns appear to be empty or have no valid data")
        
        # Remove completely empty columns (like start_time, end_time)
        empty_cols = [col for col in self.df.columns if self.df[col].isna().all()]
        if empty_cols:
            self.df = self.df.drop(columns=empty_cols)
            print(f"Removed {len(empty_cols)} empty columns: {empty_cols}")
        
        # Convert percentage win_rate to decimal if needed
        if 'win_rate' in self.df.columns and self.df['win_rate'].max() > 1:
            self.df['win_rate'] = self.df['win_rate'] / 100
        
        print(f"Data loaded: {len(self.df)} rows with {len(self.df.columns)} columns")
        print(f"Indicator columns: {self.indicator_columns}")
        print(f"Available target metrics: {available_target_metrics}")
    
    def check_available_columns(self):
        """
        Check which columns are available and provide detailed feedback
        """
        print("\n" + "="*60)
        print("COLUMN AVAILABILITY CHECK")
        print("="*60)
        
        # Check indicator columns
        indicator_groups = {
            'Moving Averages': ['ema', 'vwma', 'sma'],
            'MACD': ['macd_line', 'macd_signal'],
            'Stoch RSI': ['stoch_rsi_k', 'stoch_rsi_d', 'stoch_rsi_period'],
            'ROC': ['roc', 'roc_of_roc'],
            'Other Indicators': ['rsi', 'volatility', 'price_change', 'atr']
        }
        
        print("Indicator Groups:")
        for group_name, columns in indicator_groups.items():
            available = [col for col in columns if col in self.df.columns]
            missing = [col for col in columns if col not in self.df.columns]
            
            if available:
                print(f"  ✅ {group_name}: {available}")
            if missing:
                print(f"  ❌ {group_name} (missing): {missing}")
        
        # Check target metrics
        print("\nTarget Metrics:")
        available_targets = [col for col in self.target_metrics if col in self.df.columns]
        missing_targets = [col for col in self.target_metrics if col not in self.df.columns]
        
        if available_targets:
            print(f"  ✅ Available: {available_targets}")
        if missing_targets:
            print(f"  ❌ Missing: {missing_targets}")
        
        print("="*60)
        
        return {
            'indicator_groups': indicator_groups,
            'available_targets': available_targets,
            'missing_targets': missing_targets
        }
    
    def create_correlation_heatmap(self, figsize=(12, 8)):
        """
        Create a correlation heatmap between indicators and target metrics
        """
        # Select relevant columns for correlation (only those that exist and are numeric)
        available_target_metrics = [col for col in self.target_metrics if col in self.df.columns]
        
        # Filter out non-numeric columns
        numeric_indicator_cols = []
        for col in self.indicator_columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_indicator_cols.append(col)
        
        numeric_target_metrics = []
        for col in available_target_metrics:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_target_metrics.append(col)
        
        correlation_cols = numeric_indicator_cols + numeric_target_metrics
        
        if not numeric_target_metrics:
            print("Warning: No numeric target metrics available for correlation analysis.")
            return
        
        if not numeric_indicator_cols:
            print("Warning: No numeric indicator columns available for correlation analysis.")
            return
        
        corr_data = self.df[correlation_cols].corr()
        
        # Extract correlations between indicators and targets
        indicator_target_corr = corr_data.loc[numeric_indicator_cols, numeric_target_metrics]
        
        plt.figure(figsize=figsize)
        
        # Create custom colormap
        colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # Create heatmap
        sns.heatmap(indicator_target_corr, 
                   annot=True, 
                   cmap=cmap,
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Correlation Coefficient'},
                   annot_kws={'size': 11, 'weight': 'bold'})
        
        plt.title('Correlation: Indicator Parameters vs Performance Metrics', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Performance Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Indicator Parameters', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def create_interactive_heatmap(self, x_param, y_param, target_metric):
        """
        Create an interactive heatmap for two parameters against a target metric
        
        Parameters:
        x_param (str): Parameter for x-axis
        y_param (str): Parameter for y-axis  
        target_metric (str): Target metric to analyze
        """
        # Create pivot table
        pivot_data = self.df.pivot_table(
            values=target_metric, 
            index=y_param, 
            columns=x_param, 
            aggfunc='mean'
        )
        
        # Create interactive heatmap
        fig = px.imshow(
            pivot_data,
            labels=dict(x=x_param, y=y_param, color=target_metric),
            x=pivot_data.columns,
            y=pivot_data.index,
            aspect="auto",
            color_continuous_scale='RdYlBu_r',
            title=f'Heat Map: {target_metric.title()} vs {x_param} & {y_param}'
        )
        
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            width=800,
            height=600
        )
        
    def create_stoch_rsi_advanced_metrics_heatmaps(self):
        """
        Create heat maps for advanced trading metrics using Stoch RSI parameters
        """
        # Calculate additional metrics if they exist
        advanced_metrics = []
        metric_titles = []
        
        if 'max_drawdown' in self.df.columns:
            advanced_metrics.append('max_drawdown')
            metric_titles.append('Max Drawdown')
        
        if 'average_trade_duration (minutes)' in self.df.columns:
            advanced_metrics.append('average_trade_duration (minutes)')
            metric_titles.append('Avg Trade Duration')
        
        if 'expected_profit' in self.df.columns:
            advanced_metrics.append('expected_profit')
            metric_titles.append('Expected Profit')
        
        if not advanced_metrics:
            print("No advanced metrics found in data")
            return
        
        fig = make_subplots(
            rows=len(advanced_metrics), cols=3,
            subplot_titles=[f'{title}: Period vs K' for title in metric_titles] + 
                          [f'{title}: Period vs D' for title in metric_titles] + 
                          [f'{title}: K vs D' for title in metric_titles],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}] 
                   for _ in range(len(advanced_metrics))]
        )
        
        param_pairs = [
            ('stoch_rsi_k', 'stoch_rsi_period'),
            ('stoch_rsi_d', 'stoch_rsi_period'),
            ('stoch_rsi_d', 'stoch_rsi_k')
        ]
        
        for row, metric in enumerate(advanced_metrics):
            for col, (y_param, x_param) in enumerate(param_pairs):
                pivot = self.df.pivot_table(values=metric, index=y_param, columns=x_param, aggfunc='mean')
                fig.add_trace(
                    go.Heatmap(
                        z=pivot.values, 
                        x=pivot.columns, 
                        y=pivot.index,
                        colorscale='RdYlBu_r',
                        showscale=False
                    ),
                    row=row+1, col=col+1
                )
        
        fig.update_layout(
            title_text="Advanced Stoch RSI Metrics Analysis",
            title_x=0.5,
            title_font_size=18,
            height=400 * len(advanced_metrics),
            width=1400
        )
        
        # Update axis labels for all subplots
        for row in range(len(advanced_metrics)):
            fig.update_xaxes(title_text="Stoch RSI Period", row=row+1, col=1)
            fig.update_yaxes(title_text="Stoch RSI K", row=row+1, col=1)
            fig.update_xaxes(title_text="Stoch RSI Period", row=row+1, col=2)
            fig.update_yaxes(title_text="Stoch RSI D", row=row+1, col=2)
            fig.update_xaxes(title_text="Stoch RSI K", row=row+1, col=3)
            fig.update_yaxes(title_text="Stoch RSI D", row=row+1, col=3)
        
        fig.show()
    
    def create_stoch_rsi_ratio_heatmaps(self):
        """
        Create heat maps for ratio-based metrics using Stoch RSI parameters
        """
        # Calculate ratio metrics with safety checks
        ratio_metrics = []
        ratio_titles = []
        
        # Profit per trade ratio
        if 'total_trade_profit' in self.df.columns and 'total_trades' in self.df.columns:
            self.df['profit_per_trade_ratio'] = self.df['total_trade_profit'] / self.df['total_trades'].replace(0, 1)
            ratio_metrics.append('profit_per_trade_ratio')
            ratio_titles.append('Profit per Trade Ratio')
        
        # Win/Loss ratio
        if 'winning_trades' in self.df.columns and 'total_trades' in self.df.columns:
            # Avoid division by zero
            denominator = (self.df['total_trades'] - self.df['winning_trades']).replace(0, 1)
            self.df['win_loss_ratio'] = self.df['winning_trades'] / denominator
            ratio_metrics.append('win_loss_ratio')
            ratio_titles.append('Win/Loss Ratio')
        
        # Profit to drawdown ratio
        if 'max_drawdown' in self.df.columns and 'total_trade_profit' in self.df.columns:
            # Avoid division by zero
            denominator = abs(self.df['max_drawdown']).replace(0, 1)
            self.df['profit_to_drawdown_ratio'] = self.df['total_trade_profit'] / denominator
            ratio_metrics.append('profit_to_drawdown_ratio')
            ratio_titles.append('Profit/Drawdown Ratio')
        
        if not ratio_metrics:
            print("Warning: No ratio metrics could be calculated. Required columns missing.")
            return
        
        fig = make_subplots(
            rows=len(ratio_metrics), cols=3,
            subplot_titles=[f'{title}: Period vs K' for title in ratio_titles] + 
                          [f'{title}: Period vs D' for title in ratio_titles] + 
                          [f'{title}: K vs D' for title in ratio_titles],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}] 
                   for _ in range(len(ratio_metrics))]
        )
        
        param_pairs = [
            ('stoch_rsi_k', 'stoch_rsi_period'),
            ('stoch_rsi_d', 'stoch_rsi_period'),
            ('stoch_rsi_d', 'stoch_rsi_k')
        ]
        
        for row, metric in enumerate(ratio_metrics):
            for col, (y_param, x_param) in enumerate(param_pairs):
                pivot = self.df.pivot_table(values=metric, index=y_param, columns=x_param, aggfunc='mean')
                fig.add_trace(
                    go.Heatmap(
                        z=pivot.values, 
                        x=pivot.columns, 
                        y=pivot.index,
                        colorscale='RdYlBu_r',
                        showscale=False
                    ),
                    row=row+1, col=col+1
                )
        
        fig.update_layout(
            title_text="Stoch RSI Ratio Metrics Analysis",
            title_x=0.5,
            title_font_size=18,
            height=400 * len(ratio_metrics),
            width=1400
        )
        
        # Update axis labels for all subplots
        for row in range(len(ratio_metrics)):
            fig.update_xaxes(title_text="Stoch RSI Period", row=row+1, col=1)
            fig.update_yaxes(title_text="Stoch RSI K", row=row+1, col=1)
            fig.update_xaxes(title_text="Stoch RSI Period", row=row+1, col=2)
            fig.update_yaxes(title_text="Stoch RSI D", row=row+1, col=2)
            fig.update_xaxes(title_text="Stoch RSI K", row=row+1, col=3)
            fig.update_yaxes(title_text="Stoch RSI D", row=row+1, col=3)
        
        fig.show()
    
    def create_stoch_rsi_efficiency_heatmaps(self):
        """
        Create heat maps for trading efficiency metrics using Stoch RSI parameters
        """
        # Calculate efficiency metrics with safety checks
        efficiency_metrics = []
        efficiency_titles = []
        
        # Trades per day
        if 'total_trades' in self.df.columns and 'end_time' in self.df.columns and 'start_time' in self.df.columns:
            try:
                days_diff = (pd.to_datetime(self.df['end_time']) - pd.to_datetime(self.df['start_time'])).dt.days + 1
                days_diff = days_diff.replace(0, 1)  # Avoid division by zero
                self.df['trades_per_day'] = self.df['total_trades'] / days_diff
                efficiency_metrics.append('trades_per_day')
                efficiency_titles.append('Trades per Day')
            except Exception as e:
                print(f"Warning: Could not calculate trades_per_day: {e}")
        
        # Profit per day
        if 'total_trade_profit' in self.df.columns and 'end_time' in self.df.columns and 'start_time' in self.df.columns:
            try:
                days_diff = (pd.to_datetime(self.df['end_time']) - pd.to_datetime(self.df['start_time'])).dt.days + 1
                days_diff = days_diff.replace(0, 1)  # Avoid division by zero
                self.df['profit_per_day'] = self.df['total_trade_profit'] / days_diff
                efficiency_metrics.append('profit_per_day')
                efficiency_titles.append('Profit per Day')
            except Exception as e:
                print(f"Warning: Could not calculate profit_per_day: {e}")
        
        # Profit per minute
        if 'average_trade_profit' in self.df.columns and 'average_trade_duration (minutes)' in self.df.columns:
            try:
                duration = self.df['average_trade_duration (minutes)'].replace(0, 1)  # Avoid division by zero
                self.df['profit_per_minute'] = self.df['average_trade_profit'] / duration
                efficiency_metrics.append('profit_per_minute')
                efficiency_titles.append('Profit per Minute')
            except Exception as e:
                print(f"Warning: Could not calculate profit_per_minute: {e}")
        
        if not efficiency_metrics:
            print("Warning: No efficiency metrics could be calculated. Required columns missing.")
            return
        
        fig = make_subplots(
            rows=len(efficiency_metrics), cols=3,
            subplot_titles=[f'{title}: Period vs K' for title in efficiency_titles] + 
                          [f'{title}: Period vs D' for title in efficiency_titles] + 
                          [f'{title}: K vs D' for title in efficiency_titles],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}] 
                   for _ in range(len(efficiency_metrics))]
        )
        
        param_pairs = [
            ('stoch_rsi_k', 'stoch_rsi_period'),
            ('stoch_rsi_d', 'stoch_rsi_period'),
            ('stoch_rsi_d', 'stoch_rsi_k')
        ]
        
        for row, metric in enumerate(efficiency_metrics):
            for col, (y_param, x_param) in enumerate(param_pairs):
                pivot = self.df.pivot_table(values=metric, index=y_param, columns=x_param, aggfunc='mean')
                fig.add_trace(
                    go.Heatmap(
                        z=pivot.values, 
                        x=pivot.columns, 
                        y=pivot.index,
                        colorscale='RdYlBu_r',
                        showscale=False
                    ),
                    row=row+1, col=col+1
                )
        
        fig.update_layout(
            title_text="Stoch RSI Trading Efficiency Analysis",
            title_x=0.5,
            title_font_size=18,
            height=400 * len(efficiency_metrics),
            width=1400
        )
        
        # Update axis labels for all subplots
        for row in range(len(efficiency_metrics)):
            fig.update_xaxes(title_text="Stoch RSI Period", row=row+1, col=1)
            fig.update_yaxes(title_text="Stoch RSI K", row=row+1, col=1)
            fig.update_xaxes(title_text="Stoch RSI Period", row=row+1, col=2)
            fig.update_yaxes(title_text="Stoch RSI D", row=row+1, col=2)
            fig.update_xaxes(title_text="Stoch RSI K", row=row+1, col=3)
            fig.update_yaxes(title_text="Stoch RSI D", row=row+1, col=3)
        
        fig.show()
    
    def create_stoch_rsi_risk_heatmaps(self):
        """
        Create heat maps for risk-related metrics using Stoch RSI parameters
        """
        risk_metrics = []
        risk_titles = []
        
        if 'max_drawdown' in self.df.columns:
            risk_metrics.append('max_drawdown')
            risk_titles.append('Max Drawdown')
        
        if 'average_max_drawdown' in self.df.columns:
            risk_metrics.append('average_max_drawdown')
            risk_titles.append('Average Max Drawdown')
        
        if 'max_trade_duration (minutes)' in self.df.columns:
            risk_metrics.append('max_trade_duration (minutes)')
            risk_titles.append('Max Trade Duration')
        
        # Calculate risk-adjusted returns if possible
        if 'total_trade_profit' in self.df.columns and 'max_drawdown' in self.df.columns:
            self.df['risk_adjusted_return'] = self.df['total_trade_profit'] / abs(self.df['max_drawdown'])
            risk_metrics.append('risk_adjusted_return')
            risk_titles.append('Risk Adjusted Return')
        
        if not risk_metrics:
            print("No risk metrics found in data")
            return
        
        fig = make_subplots(
            rows=len(risk_metrics), cols=3,
            subplot_titles=[f'{title}: Period vs K' for title in risk_titles] + 
                          [f'{title}: Period vs D' for title in risk_titles] + 
                          [f'{title}: K vs D' for title in risk_titles],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}] 
                   for _ in range(len(risk_metrics))]
        )
        
        param_pairs = [
            ('stoch_rsi_k', 'stoch_rsi_period'),
            ('stoch_rsi_d', 'stoch_rsi_period'),
            ('stoch_rsi_d', 'stoch_rsi_k')
        ]
        
        for row, metric in enumerate(risk_metrics):
            for col, (y_param, x_param) in enumerate(param_pairs):
                pivot = self.df.pivot_table(values=metric, index=y_param, columns=x_param, aggfunc='mean')
                fig.add_trace(
                    go.Heatmap(
                        z=pivot.values, 
                        x=pivot.columns, 
                        y=pivot.index,
                        colorscale='RdYlBu_r',
                        showscale=False
                    ),
                    row=row+1, col=col+1
                )
        
        fig.update_layout(
            title_text="Stoch RSI Risk Metrics Analysis",
            title_x=0.5,
            title_font_size=18,
            height=400 * len(risk_metrics),
            width=1400
        )
        
        # Update axis labels for all subplots
        for row in range(len(risk_metrics)):
            fig.update_xaxes(title_text="Stoch RSI Period", row=row+1, col=1)
            fig.update_yaxes(title_text="Stoch RSI K", row=row+1, col=1)
            fig.update_xaxes(title_text="Stoch RSI Period", row=row+1, col=2)
            fig.update_yaxes(title_text="Stoch RSI D", row=row+1, col=2)
            fig.update_xaxes(title_text="Stoch RSI K", row=row+1, col=3)
            fig.update_yaxes(title_text="Stoch RSI D", row=row+1, col=3)
        
        fig.show()
    
    def create_stoch_rsi_parameter_sensitivity_analysis(self):
        """
        Create sensitivity analysis showing how much each parameter affects performance
        """
        # Calculate parameter ranges and their impact on metrics
        sensitivity_data = {}
        
        for param in ['stoch_rsi_period', 'stoch_rsi_k', 'stoch_rsi_d']:
            param_groups = self.df.groupby(param).agg({
                'win_rate': ['mean', 'std'],
                'average_trade_profit': ['mean', 'std'],
                'total_trades': ['mean', 'std']
            }).round(4)
            
            sensitivity_data[param] = param_groups
        
        # Create subplot for each parameter
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Win Rate vs Period', 'Avg Profit vs Period', 'Total Trades vs Period',
                'Win Rate vs K', 'Avg Profit vs K', 'Total Trades vs K',
                'Win Rate vs D', 'Avg Profit vs D', 'Total Trades vs D'
            ],
            specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]]
        )
        
        params = ['stoch_rsi_period', 'stoch_rsi_k', 'stoch_rsi_d']
        metrics = ['win_rate', 'average_trade_profit', 'total_trades']
        
        for row, param in enumerate(params):
            for col, metric in enumerate(metrics):
                grouped_data = self.df.groupby(param)[metric].mean().reset_index()
                
                fig.add_trace(
                    go.Scatter(
                        x=grouped_data[param],
                        y=grouped_data[metric],
                        mode='lines+markers',
                        name=f'{metric} vs {param}',
                        showlegend=False
                    ),
                    row=row+1, col=col+1
                )
        
        fig.update_layout(
            title_text="Stoch RSI Parameter Sensitivity Analysis",
            title_x=0.5,
            title_font_size=18,
            height=1200,
            width=1400
        )
        
        # Update axis labels for sensitivity analysis
        fig.update_xaxes(title_text="Stoch RSI Period", row=1, col=1)
        fig.update_yaxes(title_text="Win Rate", row=1, col=1)
        fig.update_xaxes(title_text="Stoch RSI Period", row=1, col=2)
        fig.update_yaxes(title_text="Average Trade Profit", row=1, col=2)
        fig.update_xaxes(title_text="Stoch RSI Period", row=1, col=3)
        fig.update_yaxes(title_text="Total Trades", row=1, col=3)
        
        fig.update_xaxes(title_text="Stoch RSI K", row=2, col=1)
        fig.update_yaxes(title_text="Win Rate", row=2, col=1)
        fig.update_xaxes(title_text="Stoch RSI K", row=2, col=2)
        fig.update_yaxes(title_text="Average Trade Profit", row=2, col=2)
        fig.update_xaxes(title_text="Stoch RSI K", row=2, col=3)
        fig.update_yaxes(title_text="Total Trades", row=2, col=3)
        
        fig.update_xaxes(title_text="Stoch RSI D", row=3, col=1)
        fig.update_yaxes(title_text="Win Rate", row=3, col=1)
        fig.update_xaxes(title_text="Stoch RSI D", row=3, col=2)
        fig.update_yaxes(title_text="Average Trade Profit", row=3, col=2)
        fig.update_xaxes(title_text="Stoch RSI D", row=3, col=3)
        fig.update_yaxes(title_text="Total Trades", row=3, col=3)
        
        fig.show()
        
        return sensitivity_data
    
    def create_ema_vwma_heatmaps(self):
        """
        Create EMA/VWMA heat maps for average_trade_profit, total_trades, and win_rate
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                'Average Trade Profit (EMA vs VWMA)',
                'Total Trades (EMA vs VWMA)', 
                'Win Rate (EMA vs VWMA)'
            ],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        metrics = ['average_trade_profit', 'total_trades', 'win_rate']
        
        for i, metric in enumerate(metrics):
            pivot = self.df.pivot_table(values=metric, index='vwma', columns='ema', aggfunc='mean')
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values, 
                    x=pivot.columns, 
                    y=pivot.index,
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(x=0.33*i + 0.16, len=0.3)
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title_text="EMA vs VWMA Performance Analysis",
            title_x=0.5,
            title_font_size=18,
            height=500,
            width=1400
        )
        
        # Update axis labels for all subplots
        fig.update_xaxes(title_text="EMA Period", row=1, col=1)
        fig.update_yaxes(title_text="VWMA Period", row=1, col=1)
        fig.update_xaxes(title_text="EMA Period", row=1, col=2)
        fig.update_yaxes(title_text="VWMA Period", row=1, col=2)
        fig.update_xaxes(title_text="EMA Period", row=1, col=3)
        fig.update_yaxes(title_text="VWMA Period", row=1, col=3)
        
        fig.show()
    
    def create_roc_heatmaps(self):
        """
        Create ROC heat maps for average_trade_profit, total_trades, and win_rate
        """
        # Check if ROC column exists
        if 'roc' not in self.df.columns:
            print("Warning: ROC heatmaps require 'roc' column")
            return
        
        # Check if we have ROC_of_ROC for dual-axis heatmaps
        has_roc_of_roc = 'roc_of_roc' in self.df.columns
        
        if has_roc_of_roc:
            # Create dual-axis heatmaps (ROC vs ROC_of_ROC)
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=[
                    'Average Trade Profit (ROC vs ROC_of_ROC)',
                    'Total Trades (ROC vs ROC_of_ROC)', 
                    'Win Rate (ROC vs ROC_of_ROC)'
                ],
                specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
            )
            
            metrics = ['average_trade_profit', 'total_trades', 'win_rate']
            
            for i, metric in enumerate(metrics):
                if metric in self.df.columns:
                    pivot = self.df.pivot_table(values=metric, index='roc_of_roc', columns='roc', aggfunc='mean')
                    fig.add_trace(
                        go.Heatmap(
                            z=pivot.values, 
                            x=pivot.columns, 
                            y=pivot.index,
                            colorscale='RdYlBu_r',
                            showscale=True,
                            colorbar=dict(x=0.33*i + 0.16, len=0.3)
                        ),
                        row=1, col=i+1
                    )
            
            fig.update_layout(
                title_text="ROC vs ROC_of_ROC Performance Analysis",
                title_x=0.5,
                title_font_size=18,
                height=500,
                width=1400
            )
            
            # Update axis labels for all subplots
            fig.update_xaxes(title_text="ROC Period", row=1, col=1)
            fig.update_yaxes(title_text="ROC of ROC Period", row=1, col=1)
            fig.update_xaxes(title_text="ROC Period", row=1, col=2)
            fig.update_yaxes(title_text="ROC of ROC Period", row=1, col=2)
            fig.update_xaxes(title_text="ROC Period", row=1, col=3)
            fig.update_yaxes(title_text="ROC of ROC Period", row=1, col=3)
            
            fig.show()
        else:
            # Create single-axis heatmaps (ROC only)
            print("Creating single-axis ROC heatmaps (ROC_of_ROC column not available)")
            
            # Get available metrics
            available_metrics = [metric for metric in ['average_trade_profit', 'total_trades', 'win_rate'] 
                               if metric in self.df.columns]
            
            if len(available_metrics) == 0:
                print("Warning: No available metrics for ROC heatmaps")
                return
            
            # Create subplot layout
            n_metrics = len(available_metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[f'{metric.replace("_", " ").title()} vs ROC Period' for metric in available_metrics],
                specs=[[{"type": "scatter"} for _ in range(n_cols)] for _ in range(n_rows)]
            )
            
            for i, metric in enumerate(available_metrics):
                row = (i // n_cols) + 1
                col = (i % n_cols) + 1
                
                # Group by ROC and calculate mean
                grouped = self.df.groupby('roc')[metric].mean().reset_index()
                
                fig.add_trace(
                    go.Scatter(
                        x=grouped['roc'],
                        y=grouped[metric],
                        mode='lines+markers',
                        name=metric,
                        line=dict(width=3),
                        marker=dict(size=8)
                    ),
                    row=row, col=col
                )
                
                # Update axis labels
                fig.update_xaxes(title_text="ROC Period", row=row, col=col)
                fig.update_yaxes(title_text=metric.replace("_", " ").title(), row=row, col=col)
            
            fig.update_layout(
                title_text="ROC Period Performance Analysis",
                title_x=0.5,
                title_font_size=18,
                height=300 * n_rows,
                width=400 * n_cols
            )
            
            fig.show()
    
    def create_roc_advanced_heatmaps(self):
        """
        Create advanced ROC heat maps with multiple performance metrics
        """
        # Check if ROC column exists
        if 'roc' not in self.df.columns:
            print("Warning: ROC advanced heatmaps require 'roc' column")
            return
        
        # Check if we have ROC_of_ROC for dual-axis heatmaps
        has_roc_of_roc = 'roc_of_roc' in self.df.columns
        
        # Define metrics to analyze
        metrics = ['win_rate', 'average_trade_profit', 'total_trades', 'max_drawdown', 
                  'profit_per_trade_ratio', 'profit_to_drawdown_ratio', 'trades_per_day']
        
        # Filter to only include metrics that exist in the data
        available_metrics = [metric for metric in metrics if metric in self.df.columns]
        
        if len(available_metrics) < 2:
            print("Warning: Need at least 2 available metrics for ROC advanced heatmaps")
            return
        
        if has_roc_of_roc:
            # Create dual-axis heatmaps (ROC vs ROC_of_ROC)
            # Create subplot layout
            n_metrics = len(available_metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[f'{metric.replace("_", " ").title()} (ROC vs ROC_of_ROC)' for metric in available_metrics],
                specs=[[{"type": "heatmap"} for _ in range(n_cols)] for _ in range(n_rows)]
            )
            
            for i, metric in enumerate(available_metrics):
                row = (i // n_cols) + 1
                col = (i % n_cols) + 1
                
                # Create pivot table
                pivot = self.df.pivot_table(values=metric, index='roc_of_roc', columns='roc', aggfunc='mean')
                
                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=pivot.values, 
                        x=pivot.columns, 
                        y=pivot.index,
                        colorscale='RdYlBu_r',
                        showscale=True,
                        colorbar=dict(x=0.33*(col-1) + 0.16, len=0.3),
                        name=metric
                    ),
                    row=row, col=col
                )
                
                # Update axis labels
                fig.update_xaxes(title_text="ROC Period", row=row, col=col)
                fig.update_yaxes(title_text="ROC of ROC Period", row=row, col=col)
            
            fig.update_layout(
                title_text="Advanced ROC vs ROC_of_ROC Performance Analysis",
                title_x=0.5,
                title_font_size=18,
                height=300 * n_rows,
                width=1400
            )
            
            fig.show()
        else:
            # Create single-axis advanced heatmaps (ROC only)
            print("Creating single-axis advanced ROC heatmaps (ROC_of_ROC column not available)")
            
            # Create subplot layout
            n_metrics = len(available_metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[f'{metric.replace("_", " ").title()} vs ROC Period' for metric in available_metrics],
                specs=[[{"type": "scatter"} for _ in range(n_cols)] for _ in range(n_rows)]
            )
            
            for i, metric in enumerate(available_metrics):
                row = (i // n_cols) + 1
                col = (i % n_cols) + 1
                
                # Group by ROC and calculate mean
                grouped = self.df.groupby('roc')[metric].mean().reset_index()
                
                fig.add_trace(
                    go.Scatter(
                        x=grouped['roc'],
                        y=grouped[metric],
                        mode='lines+markers',
                        name=metric,
                        line=dict(width=3),
                        marker=dict(size=8)
                    ),
                    row=row, col=col
                )
                
                # Update axis labels
                fig.update_xaxes(title_text="ROC Period", row=row, col=col)
                fig.update_yaxes(title_text=metric.replace("_", " ").title(), row=row, col=col)
            
            fig.update_layout(
                title_text="Advanced ROC Period Performance Analysis",
                title_x=0.5,
                title_font_size=18,
                height=300 * n_rows,
                width=400 * n_cols
            )
            
            fig.show()
    
    def create_roc_efficiency_heatmaps(self):
        """
        Create ROC efficiency analysis heat maps
        """
        # Check if ROC column exists
        if 'roc' not in self.df.columns:
            print("Warning: ROC efficiency heatmaps require 'roc' column")
            return
        
        # Check if we have ROC_of_ROC for dual-axis heatmaps
        has_roc_of_roc = 'roc_of_roc' in self.df.columns
        
        # Check for required efficiency metrics
        efficiency_metrics = ['profit_per_minute', 'profit_per_day', 'trades_per_day']
        available_efficiency_metrics = [metric for metric in efficiency_metrics if metric in self.df.columns]
        
        if not available_efficiency_metrics:
            print("Warning: No efficiency metrics available for ROC efficiency analysis")
            print("Available metrics for efficiency analysis: ['average_trade_duration (minutes)', 'total_trade_profit']")
            # Try alternative metrics
            alternative_metrics = ['average_trade_duration (minutes)', 'total_trade_profit']
            available_efficiency_metrics = [metric for metric in alternative_metrics if metric in self.df.columns]
            
            if not available_efficiency_metrics:
                print("No alternative metrics available either")
                return
        
        if has_roc_of_roc:
            # Create dual-axis heatmaps (ROC vs ROC_of_ROC)
            fig = make_subplots(
                rows=1, cols=len(available_efficiency_metrics),
                subplot_titles=[f'{metric.replace("_", " ").title()} Efficiency' for metric in available_efficiency_metrics],
                specs=[[{"type": "heatmap"} for _ in range(len(available_efficiency_metrics))]]
            )
            
            for i, metric in enumerate(available_efficiency_metrics):
                # Create pivot table
                pivot = self.df.pivot_table(values=metric, index='roc_of_roc', columns='roc', aggfunc='mean')
                
                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=pivot.values, 
                        x=pivot.columns, 
                        y=pivot.index,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(x=0.33*i + 0.16, len=0.3),
                        name=metric
                    ),
                    row=1, col=i+1
                )
                
                # Update axis labels
                fig.update_xaxes(title_text="ROC Period", row=1, col=i+1)
                fig.update_yaxes(title_text="ROC of ROC Period", row=1, col=i+1)
            
            fig.update_layout(
                title_text="ROC Efficiency Analysis",
                title_x=0.5,
                title_font_size=18,
                height=500,
                width=400 * len(available_efficiency_metrics)
            )
            
            fig.show()
        else:
            # Create single-axis efficiency heatmaps (ROC only)
            print("Creating single-axis ROC efficiency heatmaps (ROC_of_ROC column not available)")
            
            fig = make_subplots(
                rows=1, cols=len(available_efficiency_metrics),
                subplot_titles=[f'{metric.replace("_", " ").title()} vs ROC Period' for metric in available_efficiency_metrics],
                specs=[[{"type": "scatter"} for _ in range(len(available_efficiency_metrics))]]
            )
            
            for i, metric in enumerate(available_efficiency_metrics):
                # Group by ROC and calculate mean
                grouped = self.df.groupby('roc')[metric].mean().reset_index()
                
                fig.add_trace(
                    go.Scatter(
                        x=grouped['roc'],
                        y=grouped[metric],
                        mode='lines+markers',
                        name=metric,
                        line=dict(width=3),
                        marker=dict(size=8)
                    ),
                    row=1, col=i+1
                )
                
                # Update axis labels
                fig.update_xaxes(title_text="ROC Period", row=1, col=i+1)
                fig.update_yaxes(title_text=metric.replace("_", " ").title(), row=1, col=i+1)
            
            fig.update_layout(
                title_text="ROC Efficiency Analysis",
                title_x=0.5,
                title_font_size=18,
                height=500,
                width=400 * len(available_efficiency_metrics)
            )
            
            fig.show()
    
    def create_roc_risk_heatmaps(self):
        """
        Create ROC risk analysis heat maps
        """
        # Check if ROC column exists
        if 'roc' not in self.df.columns:
            print("Warning: ROC risk heatmaps require 'roc' column")
            return
        
        # Check if we have ROC_of_ROC for dual-axis heatmaps
        has_roc_of_roc = 'roc_of_roc' in self.df.columns
        
        # Check for required risk metrics
        risk_metrics = ['max_drawdown', 'win_loss_ratio', 'profit_to_drawdown_ratio']
        available_risk_metrics = [metric for metric in risk_metrics if metric in self.df.columns]
        
        if not available_risk_metrics:
            print("Warning: No risk metrics available for ROC risk analysis")
            print("Available metrics for risk analysis: ['max_drawdown', 'win_rate']")
            # Try alternative metrics
            alternative_metrics = ['max_drawdown', 'win_rate']
            available_risk_metrics = [metric for metric in alternative_metrics if metric in self.df.columns]
            
            if not available_risk_metrics:
                print("No alternative risk metrics available either")
                return
        
        if has_roc_of_roc:
            # Create dual-axis heatmaps (ROC vs ROC_of_ROC)
            fig = make_subplots(
                rows=1, cols=len(available_risk_metrics),
                subplot_titles=[f'{metric.replace("_", " ").title()} Risk Analysis' for metric in available_risk_metrics],
                specs=[[{"type": "heatmap"} for _ in range(len(available_risk_metrics))]]
            )
            
            for i, metric in enumerate(available_risk_metrics):
                # Create pivot table
                pivot = self.df.pivot_table(values=metric, index='roc_of_roc', columns='roc', aggfunc='mean')
                
                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=pivot.values, 
                        x=pivot.columns, 
                        y=pivot.index,
                        colorscale='Reds' if 'drawdown' in metric else 'Greens',
                        showscale=True,
                        colorbar=dict(x=0.33*i + 0.16, len=0.3),
                        name=metric
                    ),
                    row=1, col=i+1
                )
                
                # Update axis labels
                fig.update_xaxes(title_text="ROC Period", row=1, col=i+1)
                fig.update_yaxes(title_text="ROC of ROC Period", row=1, col=i+1)
            
            fig.update_layout(
                title_text="ROC Risk Analysis",
                title_x=0.5,
                title_font_size=18,
                height=500,
                width=400 * len(available_risk_metrics)
            )
            
            fig.show()
        else:
            # Create single-axis risk heatmaps (ROC only)
            print("Creating single-axis ROC risk heatmaps (ROC_of_ROC column not available)")
            
            fig = make_subplots(
                rows=1, cols=len(available_risk_metrics),
                subplot_titles=[f'{metric.replace("_", " ").title()} vs ROC Period' for metric in available_risk_metrics],
                specs=[[{"type": "scatter"} for _ in range(len(available_risk_metrics))]]
            )
            
            for i, metric in enumerate(available_risk_metrics):
                # Group by ROC and calculate mean
                grouped = self.df.groupby('roc')[metric].mean().reset_index()
                
                fig.add_trace(
                    go.Scatter(
                        x=grouped['roc'],
                        y=grouped[metric],
                        mode='lines+markers',
                        name=metric,
                        line=dict(width=3),
                        marker=dict(size=8)
                    ),
                    row=1, col=i+1
                )
                
                # Update axis labels
                fig.update_xaxes(title_text="ROC Period", row=1, col=i+1)
                fig.update_yaxes(title_text=metric.replace("_", " ").title(), row=1, col=i+1)
            
            fig.update_layout(
                title_text="ROC Risk Analysis",
                title_x=0.5,
                title_font_size=18,
                height=500,
                width=400 * len(available_risk_metrics)
            )
            
            fig.show()
    
    def create_roc_vs_other_indicators_heatmaps(self):
        """
        Create ROC vs other available indicators heat maps
        """
        # Check if ROC column exists
        if 'roc' not in self.df.columns:
            print("Warning: ROC vs other indicators heatmaps require 'roc' column")
            return
        
        # Find other available indicators
        other_indicators = ['ema', 'vwma', 'stoch_rsi_period', 'stoch_rsi_k', 'stoch_rsi_d', 
                           'macd_fast', 'macd_slow', 'macd_signal']
        available_indicators = [ind for ind in other_indicators if ind in self.df.columns]
        
        if not available_indicators:
            print("Warning: No other indicators available for ROC comparison")
            return
        
        # Get available performance metrics
        performance_metrics = ['win_rate', 'average_trade_profit', 'total_trades', 'max_drawdown']
        available_metrics = [metric for metric in performance_metrics if metric in self.df.columns]
        
        if not available_metrics:
            print("Warning: No performance metrics available for ROC vs indicators analysis")
            return
        
        print(f"Creating ROC vs {len(available_indicators)} other indicators heatmaps")
        print(f"Available indicators: {available_indicators}")
        print(f"Available metrics: {available_metrics}")
        
        # Create heatmaps for each indicator vs ROC
        for indicator in available_indicators:
            fig = make_subplots(
                rows=1, cols=len(available_metrics),
                subplot_titles=[f'{metric.replace("_", " ").title()} (ROC vs {indicator.upper()})' 
                              for metric in available_metrics],
                specs=[[{"type": "heatmap"} for _ in range(len(available_metrics))]]
            )
            
            for i, metric in enumerate(available_metrics):
                # Create pivot table
                pivot = self.df.pivot_table(values=metric, index=indicator, columns='roc', aggfunc='mean')
                
                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=pivot.values, 
                        x=pivot.columns, 
                        y=pivot.index,
                        colorscale='RdYlBu_r',
                        showscale=True,
                        colorbar=dict(x=0.33*i + 0.16, len=0.3),
                        name=metric
                    ),
                    row=1, col=i+1
                )
                
                # Update axis labels
                fig.update_xaxes(title_text="ROC Period", row=1, col=i+1)
                fig.update_yaxes(title_text=f"{indicator.upper()} Period", row=1, col=i+1)
            
            fig.update_layout(
                title_text=f"ROC vs {indicator.upper()} Performance Analysis",
                title_x=0.5,
                title_font_size=18,
                height=500,
                width=400 * len(available_metrics)
            )
            
            fig.show()
    
    def create_stoch_rsi_period_heatmaps(self):
        """
        Create Stoch RSI Period heat maps for average_trade_profit, total_trades, and win_rate
        Note: This creates period vs k since you mentioned "stoch rsi for average trades, total trades and win rate"
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                'Average Trade Profit (Stoch RSI Period vs K)',
                'Total Trades (Stoch RSI Period vs K)', 
                'Win Rate (Stoch RSI Period vs K)'
            ],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        metrics = ['average_trade_profit', 'total_trades', 'win_rate']
        
        for i, metric in enumerate(metrics):
            pivot = self.df.pivot_table(values=metric, index='stoch_rsi_k', columns='stoch_rsi_period', aggfunc='mean')
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values, 
                    x=pivot.columns, 
                    y=pivot.index,
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(x=0.33*i + 0.16, len=0.3)
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title_text="Stoch RSI Period vs K Performance Analysis",
            title_x=0.5,
            title_font_size=18,
            height=500,
            width=1400
        )
        
        # Update axis labels for all subplots
        fig.update_xaxes(title_text="Stoch RSI Period", row=1, col=1)
        fig.update_yaxes(title_text="Stoch RSI K", row=1, col=1)
        fig.update_xaxes(title_text="Stoch RSI Period", row=1, col=2)
        fig.update_yaxes(title_text="Stoch RSI K", row=1, col=2)
        fig.update_xaxes(title_text="Stoch RSI Period", row=1, col=3)
        fig.update_yaxes(title_text="Stoch RSI K", row=1, col=3)
        
        fig.show()
    
    def create_stoch_k_d_heatmaps(self):
        """
        Create Stoch RSI K vs D heat maps for average_trade_profit, total_trades, and win_rate
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                'Average Trade Profit (Stoch K vs D)',
                'Total Trades (Stoch K vs D)', 
                'Win Rate (Stoch K vs D)'
            ],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        metrics = ['average_trade_profit', 'total_trades', 'win_rate']
        
        for i, metric in enumerate(metrics):
            pivot = self.df.pivot_table(values=metric, index='stoch_rsi_d', columns='stoch_rsi_k', aggfunc='mean')
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values, 
                    x=pivot.columns, 
                    y=pivot.index,
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(x=0.33*i + 0.16, len=0.3)
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title_text="Stoch RSI K vs D Performance Analysis",
            title_x=0.5,
            title_font_size=18,
            height=500,
            width=1400
        )
        
        # Update axis labels for all subplots
        fig.update_xaxes(title_text="Stoch RSI K", row=1, col=1)
        fig.update_yaxes(title_text="Stoch RSI D", row=1, col=1)
        fig.update_xaxes(title_text="Stoch RSI K", row=1, col=2)
        fig.update_yaxes(title_text="Stoch RSI D", row=1, col=2)
        fig.update_xaxes(title_text="Stoch RSI K", row=1, col=3)
        fig.update_yaxes(title_text="Stoch RSI D", row=1, col=3)
        
        fig.show()
    
    def create_stoch_rsi_3d_heatmaps(self):
        """
        Create 3D analysis for Stoch RSI Period, K, and D parameters
        This creates multiple 2D views since true 3D heatmaps are complex
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Avg Profit: Period vs K', 'Avg Profit: Period vs D', 'Avg Profit: K vs D',
                'Total Trades: Period vs K', 'Total Trades: Period vs D', 'Total Trades: K vs D', 
                'Win Rate: Period vs K', 'Win Rate: Period vs D', 'Win Rate: K vs D'
            ],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        metrics = ['average_trade_profit', 'total_trades', 'win_rate']
        param_pairs = [
            ('stoch_rsi_k', 'stoch_rsi_period'),
            ('stoch_rsi_d', 'stoch_rsi_period'),
            ('stoch_rsi_d', 'stoch_rsi_k')
        ]
        
        for row, metric in enumerate(metrics):
            for col, (y_param, x_param) in enumerate(param_pairs):
                pivot = self.df.pivot_table(values=metric, index=y_param, columns=x_param, aggfunc='mean')
                fig.add_trace(
                    go.Heatmap(
                        z=pivot.values, 
                        x=pivot.columns, 
                        y=pivot.index,
                        colorscale='RdYlBu_r',
                        showscale=False
                    ),
                    row=row+1, col=col+1
                )
        
        fig.update_layout(
            title_text="Comprehensive Stoch RSI Analysis (Period, K, D)",
            title_x=0.5,
            title_font_size=18,
            height=1200,
            width=1400
        )
        
        # Update axis labels for all subplots
        # Row 1: Average Profit
        fig.update_xaxes(title_text="Stoch RSI Period", row=1, col=1)
        fig.update_yaxes(title_text="Stoch RSI K", row=1, col=1)
        fig.update_xaxes(title_text="Stoch RSI Period", row=1, col=2)
        fig.update_yaxes(title_text="Stoch RSI D", row=1, col=2)
        fig.update_xaxes(title_text="Stoch RSI K", row=1, col=3)
        fig.update_yaxes(title_text="Stoch RSI D", row=1, col=3)
        
        # Row 2: Total Trades
        fig.update_xaxes(title_text="Stoch RSI Period", row=2, col=1)
        fig.update_yaxes(title_text="Stoch RSI K", row=2, col=1)
        fig.update_xaxes(title_text="Stoch RSI Period", row=2, col=2)
        fig.update_yaxes(title_text="Stoch RSI D", row=2, col=2)
        fig.update_xaxes(title_text="Stoch RSI K", row=2, col=3)
        fig.update_yaxes(title_text="Stoch RSI D", row=2, col=3)
        
        # Row 3: Win Rate
        fig.update_xaxes(title_text="Stoch RSI Period", row=3, col=1)
        fig.update_yaxes(title_text="Stoch RSI K", row=3, col=1)
        fig.update_xaxes(title_text="Stoch RSI Period", row=3, col=2)
        fig.update_yaxes(title_text="Stoch RSI D", row=3, col=2)
        fig.update_xaxes(title_text="Stoch RSI K", row=3, col=3)
        fig.update_yaxes(title_text="Stoch RSI D", row=3, col=3)
        
        fig.show()
    
    def find_optimal_parameters(self, target_metric='win_rate', top_n=10):
        """
        Find the top parameter combinations for a target metric
        
        Parameters:
        target_metric (str): Metric to optimize for
        top_n (int): Number of top combinations to return
        """
        # Sort by target metric and get top combinations
        top_combinations = self.df.nlargest(top_n, target_metric)
        
        result_cols = self.indicator_columns + [target_metric, 'total_trades', 'average_trade_profit']
        result_df = top_combinations[result_cols].round(4)
        
        print(f"\nTop {top_n} Parameter Combinations for {target_metric.title()}:")
        print("=" * 80)
        print(result_df.to_string(index=False))
        
        return result_df
    
    def create_parameter_distribution_plots(self):
        """
        Create distribution plots for each indicator parameter
        """
        # Calculate the optimal grid size based on number of parameters
        num_params = len(self.indicator_columns)
        
        if num_params == 0:
            print("Warning: No indicator parameters available for distribution plots.")
            return
        
        # Calculate grid dimensions
        if num_params <= 4:
            rows, cols = 1, num_params
        elif num_params <= 8:
            rows, cols = 2, 4
        else:
            rows = (num_params + 3) // 4  # Ceiling division
            cols = 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        
        # Handle single subplot case
        if num_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Create plots for each parameter
        for i, param in enumerate(self.indicator_columns):
            if i < len(axes):
                axes[i].hist(self.df[param], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribution of {param}', fontweight='bold')
                axes[i].set_xlabel(param)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Distribution of Indicator Parameters', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def performance_summary(self):
        """
        Print a summary of the dataset performance metrics
        """
        print("\n" + "="*60)
        print("TRADING STRATEGY PERFORMANCE SUMMARY")
        print("="*60)
        
        summary_stats = {
            'Total Configurations': len(self.df),
            'Average Win Rate': f"{self.df['win_rate'].mean():.2%}",
            'Best Win Rate': f"{self.df['win_rate'].max():.2%}",
            'Average Profit per Trade': f"${self.df['average_trade_profit'].mean():.2f}",
            'Best Average Profit': f"${self.df['average_trade_profit'].max():.2f}",
            'Average Total Trades': f"{self.df['total_trades'].mean():.0f}",
            'Max Total Trades': f"{self.df['total_trades'].max():.0f}"
        }
        
        for key, value in summary_stats.items():
            print(f"{key:.<25} {value}")
        
        print("="*60)

# Example usage
def main():
    """
    Main function demonstrating how to use the TradingHeatMapAnalyzer
    """
    # Initialize analyzer (replace with your CSV file path)
    csv_file_path = "data/analysis/SPY.csv"  # Replace with your actual file path
    
    try:
        analyzer = TradingHeatMapAnalyzer(csv_file_path)
        
        # Check available columns first
        print("\n0. Checking available columns...")
        column_info = analyzer.check_available_columns()
        
        # Print performance summary
        analyzer.performance_summary()
        
        # Create correlation heatmap
        print("\n1. Creating correlation heatmap...")
        analyzer.create_correlation_heatmap()
        
        # Create parameter distribution plots
        print("\n2. Creating parameter distribution plots...")
        analyzer.create_parameter_distribution_plots()
        
        # Create specific heat map combinations you requested
        print("\n3. Creating EMA/VWMA heat maps...")
        if 'ema' in analyzer.df.columns and 'vwma' in analyzer.df.columns:
            analyzer.create_ema_vwma_heatmaps()
        else:
            print("EMA/VWMA heat maps not created because 'ema' or 'vwma' columns are missing.")
        
        print("\n4. Creating ROC heat maps...")
        if 'roc' in analyzer.df.columns:
            analyzer.create_roc_heatmaps()
            # Only create interactive heatmap if both ROC and ROC_of_ROC exist
            if 'roc_of_roc' in analyzer.df.columns:
                analyzer.create_interactive_heatmap('roc', 'roc_of_roc', 'average_trade_profit')
            else:
                print("Interactive ROC heatmap not created because 'roc_of_roc' column is missing.")
        else:
            print("ROC heat maps not created because 'roc' column is missing.")
        
        print("\n4a. Creating advanced ROC heat maps...")
        if 'roc' in analyzer.df.columns:
            analyzer.create_roc_advanced_heatmaps()
        else:
            print("Advanced ROC heat maps not created because 'roc' column is missing.")
        
        print("\n4b. Creating ROC efficiency heat maps...")
        if 'roc' in analyzer.df.columns:
            analyzer.create_roc_efficiency_heatmaps()
        else:
            print("ROC efficiency heat maps not created because 'roc' column is missing.")
        
        print("\n4c. Creating ROC risk heat maps...")
        if 'roc' in analyzer.df.columns:
            analyzer.create_roc_risk_heatmaps()
        else:
            print("ROC risk heat maps not created because 'roc' column is missing.")
        
        print("\n4d. Creating ROC vs other indicators heat maps...")
        if 'roc' in analyzer.df.columns:
            analyzer.create_roc_vs_other_indicators_heatmaps()
        else:
            print("ROC vs other indicators heat maps not created because 'roc' column is missing.")
        
        print("\n5. Creating Stoch RSI Period heat maps...")
        if all(col in analyzer.df.columns for col in ['stoch_rsi_period', 'stoch_rsi_k', 'stoch_rsi_d']):
            analyzer.create_stoch_rsi_period_heatmaps()
        else:
            print("Stoch RSI Period heat maps not created because required Stoch RSI columns are missing.")
        
        print("\n6. Creating Stoch RSI K vs D heat maps...")
        if all(col in analyzer.df.columns for col in ['stoch_rsi_k', 'stoch_rsi_d']):
            analyzer.create_stoch_k_d_heatmaps()
        else:
            print("Stoch RSI K vs D heat maps not created because required Stoch RSI columns are missing.")
        
        print("\n7. Creating comprehensive Stoch RSI analysis (Period, K, D)...")
        if all(col in analyzer.df.columns for col in ['stoch_rsi_period', 'stoch_rsi_k', 'stoch_rsi_d']):
            analyzer.create_stoch_rsi_3d_heatmaps()
        else:
            print("Comprehensive Stoch RSI analysis not created because required Stoch RSI columns are missing.")
        
        print("\n8. Creating advanced Stoch RSI metrics heat maps...")
        if all(col in analyzer.df.columns for col in ['stoch_rsi_period', 'stoch_rsi_k', 'stoch_rsi_d']):
            analyzer.create_stoch_rsi_advanced_metrics_heatmaps()
        else:
            print("Advanced Stoch RSI metrics heat maps not created because required Stoch RSI columns are missing.")
        
        print("\n9. Creating Stoch RSI ratio analysis heat maps...")
        if all(col in analyzer.df.columns for col in ['stoch_rsi_period', 'stoch_rsi_k', 'stoch_rsi_d']):
            analyzer.create_stoch_rsi_ratio_heatmaps()
        else:
            print("Stoch RSI ratio analysis heat maps not created because required Stoch RSI columns are missing.")
        
        print("\n10. Creating Stoch RSI efficiency analysis heat maps...")
        if all(col in analyzer.df.columns for col in ['stoch_rsi_period', 'stoch_rsi_k', 'stoch_rsi_d']):
            analyzer.create_stoch_rsi_efficiency_heatmaps()
        else:
            print("Stoch RSI efficiency analysis heat maps not created because required Stoch RSI columns are missing.")
        
        print("\n11. Creating Stoch RSI risk analysis heat maps...")
        analyzer.create_stoch_rsi_risk_heatmaps()
        
        print("\n12. Creating Stoch RSI parameter sensitivity analysis...")
        sensitivity_data = analyzer.create_stoch_rsi_parameter_sensitivity_analysis()
        
        # Find optimal parameters for different metrics
        print("\n13. Finding optimal parameters...")
        analyzer.find_optimal_parameters('win_rate', top_n=5)
        analyzer.find_optimal_parameters('average_trade_profit', top_n=5)
        analyzer.find_optimal_parameters('total_trades', top_n=5)
        
        # Create individual interactive heatmaps (optional)
        print("\n14. Creating additional individual interactive heatmaps...")
        analyzer.create_interactive_heatmap('ema', 'vwma', 'win_rate')
        
        
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file at '{csv_file_path}'")
        print("Please update the csv_file_path variable with the correct path to your data file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()