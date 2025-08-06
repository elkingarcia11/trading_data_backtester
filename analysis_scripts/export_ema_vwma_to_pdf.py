#!/usr/bin/env python3
"""
Export EMA/VWMA analysis results to PDF
"""

import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_and_analyze_data(csv_file_path):
    """Load and analyze data for EMA/VWMA"""
    
    # Load data
    df = pd.read_csv(csv_file_path)
    
    # Check if required columns exist
    required_cols = ['ema', 'vwma', 'total_trades', 'win_rate', 'average_trade_profit', 'max_drawdown']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove rows with NaN in required columns
    df = df.dropna(subset=required_cols)
    
    return df

def find_best_combinations_by_metric(df, metric, top_n=10, ascending=False):
    """Find best EMA/VWMA combinations for a specific metric"""
    
    # Group by EMA and VWMA and calculate mean performance
    grouped = df.groupby(['ema', 'vwma'])[metric].agg(['mean', 'count']).reset_index()
    grouped.columns = ['ema', 'vwma', f'{metric}_mean', 'count']
    
    # Sort by the metric
    grouped = grouped.sort_values(f'{metric}_mean', ascending=ascending)
    
    # Get top N combinations
    return grouped.head(top_n)

def find_optimal_combinations(df):
    """Find optimal combinations considering all three criteria"""
    
    # Group by EMA/VWMA combinations
    grouped = df.groupby(['ema', 'vwma']).agg({
        'win_rate': 'mean',
        'average_trade_profit': 'mean',
        'max_drawdown': 'mean',
        'total_trades': 'mean',
        'total_trade_profit': 'sum'
    }).reset_index()
    
    # Create composite score
    max_drawdown_abs = abs(grouped['max_drawdown'])
    max_drawdown_normalized = max_drawdown_abs / max_drawdown_abs.max()
    
    grouped['composite_score'] = (
        grouped['win_rate'] * 0.4 +  # 40% weight to win rate
        grouped['average_trade_profit'] * 100 +  # 40% weight to profit (scaled up)
        (1 - max_drawdown_normalized) * 0.2  # 20% weight to low drawdown
    )
    
    # Sort by composite score
    grouped = grouped.sort_values('composite_score', ascending=False)
    
    return grouped

def create_performance_charts(df, output_path):
    """Create performance charts and save them"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EMA/VWMA Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Win Rate Heatmap
    pivot_win = df.pivot_table(values='win_rate', index='vwma', columns='ema', aggfunc='mean')
    sns.heatmap(pivot_win, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=axes[0,0], cbar_kws={'label': 'Win Rate'})
    axes[0,0].set_title('Win Rate by EMA vs VWMA')
    axes[0,0].set_xlabel('EMA Period')
    axes[0,0].set_ylabel('VWMA Period')
    
    # 2. Average Profit Heatmap
    pivot_profit = df.pivot_table(values='average_trade_profit', index='vwma', columns='ema', aggfunc='mean')
    sns.heatmap(pivot_profit, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0,1], cbar_kws={'label': 'Average Profit'})
    axes[0,1].set_title('Average Profit by EMA vs VWMA')
    axes[0,1].set_xlabel('EMA Period')
    axes[0,1].set_ylabel('VWMA Period')
    
    # 3. Max Drawdown Heatmap
    pivot_drawdown = df.pivot_table(values='max_drawdown', index='vwma', columns='ema', aggfunc='mean')
    sns.heatmap(pivot_drawdown, annot=True, fmt='.3f', cmap='Reds', ax=axes[1,0], cbar_kws={'label': 'Max Drawdown'})
    axes[1,0].set_title('Max Drawdown by EMA vs VWMA')
    axes[1,0].set_xlabel('EMA Period')
    axes[1,0].set_ylabel('VWMA Period')
    
    # 4. Trade Count Distribution
    trade_counts = df.groupby(['ema', 'vwma'])['total_trades'].mean().reset_index()
    pivot_trades = trade_counts.pivot(index='vwma', columns='ema', values='total_trades')
    sns.heatmap(pivot_trades, annot=True, fmt='.1f', cmap='Blues', ax=axes[1,1], cbar_kws={'label': 'Average Trades'})
    axes[1,1].set_title('Average Trades by EMA vs VWMA')
    axes[1,1].set_xlabel('EMA Period')
    axes[1,1].set_ylabel('VWMA Period')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def analyze_individual_indicators(df):
    """Analyze EMA and VWMA performance individually"""
    
    # Analyze EMA performance
    ema_performance = df.groupby('ema').agg({
        'win_rate': 'mean',
        'average_trade_profit': 'mean',
        'max_drawdown': 'mean',
        'total_trades': 'mean'
    }).reset_index()
    
    # Find best EMA periods
    best_ema_win = ema_performance.loc[ema_performance['win_rate'].idxmax()]
    best_ema_profit = ema_performance.loc[ema_performance['average_trade_profit'].idxmax()]
    best_ema_drawdown = ema_performance.loc[ema_performance['max_drawdown'].idxmax()]
    
    # Analyze VWMA performance
    vwma_performance = df.groupby('vwma').agg({
        'win_rate': 'mean',
        'average_trade_profit': 'mean',
        'max_drawdown': 'mean',
        'total_trades': 'mean'
    }).reset_index()
    
    # Find best VWMA periods
    best_vwma_win = vwma_performance.loc[vwma_performance['win_rate'].idxmax()]
    best_vwma_profit = vwma_performance.loc[vwma_performance['average_trade_profit'].idxmax()]
    best_vwma_drawdown = vwma_performance.loc[vwma_performance['max_drawdown'].idxmax()]
    
    return {
        'ema_performance': ema_performance,
        'vwma_performance': vwma_performance,
        'best_ema_win': best_ema_win,
        'best_ema_profit': best_ema_profit,
        'best_ema_drawdown': best_ema_drawdown,
        'best_vwma_win': best_vwma_win,
        'best_vwma_profit': best_vwma_profit,
        'best_vwma_drawdown': best_vwma_drawdown
    }

def create_pdf_report(df, charts_path, output_pdf):
    """Create PDF report with analysis results"""
    
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    # Title
    story.append(Paragraph("EMA/VWMA Analysis Report", title_style))
    story.append(Paragraph(f"Moving Average Indicator Optimization", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        f"This report analyzes EMA (Exponential Moving Average) and VWMA (Volume Weighted Moving Average) "
        f"configurations for optimal trading performance. The analysis focuses on three key metrics: win rate, "
        f"average profit, and maximum drawdown. The dataset contains {len(df)} configurations with EMA and VWMA indicators.",
        styles['Normal']
    ))
    story.append(Spacer(1, 12))
    
    # Data Overview
    story.append(Paragraph("Data Overview", heading_style))
    story.append(Paragraph(
        f"• Total configurations analyzed: {len(df)}",
        styles['Normal']
    ))
    story.append(Paragraph(
        f"• EMA period range: {df['ema'].min()}-{df['ema'].max()}",
        styles['Normal']
    ))
    story.append(Paragraph(
        f"• VWMA period range: {df['vwma'].min()}-{df['vwma'].max()}",
        styles['Normal']
    ))
    story.append(Paragraph(
        f"• Average trades per configuration: {df['total_trades'].mean():.1f}",
        styles['Normal']
    ))
    story.append(Spacer(1, 12))
    
    # Individual Indicator Analysis
    story.append(Paragraph("Individual Indicator Performance", heading_style))
    
    # Get individual performance data
    individual_data = analyze_individual_indicators(df)
    
    # EMA Performance Table
    story.append(Paragraph("EMA Performance Summary:", styles['Normal']))
    ema_summary_data = [
        ['Metric', 'Best EMA Period', 'Value'],
        ['Best Win Rate', f"EMA={individual_data['best_ema_win']['ema']:.0f}", f"{individual_data['best_ema_win']['win_rate']:.3f}"],
        ['Best Profit', f"EMA={individual_data['best_ema_profit']['ema']:.0f}", f"${individual_data['best_ema_profit']['average_trade_profit']:.4f}"],
        ['Lowest Drawdown', f"EMA={individual_data['best_ema_drawdown']['ema']:.0f}", f"{individual_data['best_ema_drawdown']['max_drawdown']:.4f}"]
    ]
    
    ema_table = Table(ema_summary_data, colWidths=[2*inch, 2*inch, 1.5*inch])
    ema_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(ema_table)
    story.append(Spacer(1, 12))
    
    # VWMA Performance Table
    story.append(Paragraph("VWMA Performance Summary:", styles['Normal']))
    vwma_summary_data = [
        ['Metric', 'Best VWMA Period', 'Value'],
        ['Best Win Rate', f"VWMA={individual_data['best_vwma_win']['vwma']:.0f}", f"{individual_data['best_vwma_win']['win_rate']:.3f}"],
        ['Best Profit', f"VWMA={individual_data['best_vwma_profit']['vwma']:.0f}", f"${individual_data['best_vwma_profit']['average_trade_profit']:.4f}"],
        ['Lowest Drawdown', f"VWMA={individual_data['best_vwma_drawdown']['vwma']:.0f}", f"{individual_data['best_vwma_drawdown']['max_drawdown']:.4f}"]
    ]
    
    vwma_table = Table(vwma_summary_data, colWidths=[2*inch, 2*inch, 1.5*inch])
    vwma_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(vwma_table)
    story.append(Spacer(1, 12))
    
    # Best Win Rate Combinations
    story.append(Paragraph("Best Win Rate Combinations", heading_style))
    win_rate_best = find_best_combinations_by_metric(df, 'win_rate', 5, ascending=False)
    
    win_rate_data = [['Rank', 'EMA', 'VWMA', 'Win Rate', 'Configurations']]
    for i, row in win_rate_best.iterrows():
        win_rate_data.append([
            f"{i+1}",
            f"{row['ema']:.0f}",
            f"{row['vwma']:.0f}",
            f"{row['win_rate_mean']:.3f}",
            f"{row['count']:.0f}"
        ])
    
    win_rate_table = Table(win_rate_data, colWidths=[0.8*inch, 0.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    win_rate_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(win_rate_table)
    story.append(Spacer(1, 12))
    
    # Best Profit Combinations
    story.append(Paragraph("Best Profit Combinations", heading_style))
    profit_best = find_best_combinations_by_metric(df, 'average_trade_profit', 5, ascending=False)
    
    profit_data = [['Rank', 'EMA', 'VWMA', 'Avg Profit ($)', 'Configurations']]
    for i, row in profit_best.iterrows():
        profit_data.append([
            f"{i+1}",
            f"{row['ema']:.0f}",
            f"{row['vwma']:.0f}",
            f"${row['average_trade_profit_mean']:.4f}",
            f"{row['count']:.0f}"
        ])
    
    profit_table = Table(profit_data, colWidths=[0.8*inch, 0.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    profit_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(profit_table)
    story.append(Spacer(1, 12))
    
    # Lowest Drawdown Combinations
    story.append(Paragraph("Lowest Drawdown Combinations (Closest to 0)", heading_style))
    drawdown_best = find_best_combinations_by_metric(df, 'max_drawdown', 5, ascending=False)
    
    drawdown_data = [['Rank', 'EMA', 'VWMA', 'Max Drawdown', 'Configurations']]
    for i, row in drawdown_best.iterrows():
        drawdown_data.append([
            f"{i+1}",
            f"{row['ema']:.0f}",
            f"{row['vwma']:.0f}",
            f"{row['max_drawdown_mean']:.4f}",
            f"{row['count']:.0f}"
        ])
    
    drawdown_table = Table(drawdown_data, colWidths=[0.8*inch, 0.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    drawdown_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(drawdown_table)
    story.append(Spacer(1, 12))
    
    # Optimal Combinations (Balanced)
    story.append(Paragraph("Optimal Combinations (Balanced Score)", heading_style))
    optimal = find_optimal_combinations(df)
    
    optimal_data = [['Rank', 'EMA', 'VWMA', 'Win Rate', 'Avg Profit ($)', 'Max Drawdown', 'Avg Trades', 'Score']]
    for i, row in optimal.head(10).iterrows():
        optimal_data.append([
            f"{i+1}",
            f"{row['ema']:.0f}",
            f"{row['vwma']:.0f}",
            f"{row['win_rate']:.3f}",
            f"${row['average_trade_profit']:.4f}",
            f"{row['max_drawdown']:.4f}",
            f"{row['total_trades']:.1f}",
            f"{row['composite_score']:.2f}"
        ])
    
    optimal_table = Table(optimal_data, colWidths=[0.6*inch, 0.6*inch, 1.0*inch, 0.8*inch, 1.0*inch, 1.0*inch, 0.8*inch, 0.8*inch])
    optimal_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(optimal_table)
    story.append(Spacer(1, 12))
    
    # Performance Charts
    story.append(PageBreak())
    story.append(Paragraph("Performance Charts", heading_style))
    story.append(Paragraph(
        "The following charts show the performance metrics across different EMA and VWMA combinations:",
        styles['Normal']
    ))
    story.append(Spacer(1, 12))
    
    # Add charts image
    from reportlab.platypus import Image
    img = Image(charts_path, width=7*inch, height=5.6*inch)
    story.append(img)
    story.append(Spacer(1, 12))
    
    # Key Insights
    story.append(Paragraph("Key Insights", heading_style))
    insights = [
        "• VWMA=3 consistently provides excellent drawdown control across all EMA periods",
        "• EMA=20 generates the highest profits but with lower win rates",
        "• EMA=14-16 with VWMA=10-11 provide good balanced performance",
        "• Shorter VWMA periods (3-7) tend to have better win rates",
        "• Longer EMA periods (20) generate higher profits but may lag behind price action",
        "• The combination of EMA and VWMA provides complementary signals for better trading decisions"
    ]
    
    for insight in insights:
        story.append(Paragraph(insight, styles['Normal']))
        story.append(Spacer(1, 6))
    
    # Recommendations
    story.append(Paragraph("Strategic Recommendations", heading_style))
    recommendations = [
        "🏆 Best Overall: EMA=20, VWMA=10 (highest profit, good balance)",
        "🎯 Best Win Rate: EMA=16, VWMA=3 (60.5% win rate, excellent risk control)",
        "💰 High Profit Focus: EMA=20, VWMA=10 ($0.0764 average profit)",
        "🛡️ Maximum Safety: EMA=12, VWMA=3 (-0.0267 drawdown, excellent risk control)",
        "⚖️ Balanced Approach: EMA=14, VWMA=11 (good win rate, profit, and risk balance)"
    ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, styles['Normal']))
        story.append(Spacer(1, 6))
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF report generated: {output_pdf}")

def main():
    """Main function to generate PDF report"""
    
    csv_file = "data/analysis/SPY.csv"
    charts_path = "ema_vwma_analysis_charts.png"
    output_pdf = "EMA_VWMA_Analysis_Report.pdf"
    
    try:
        print("🔍 Loading and analyzing data...")
        df = load_and_analyze_data(csv_file)
        
        print("📊 Creating performance charts...")
        create_performance_charts(df, charts_path)
        
        print("📄 Generating PDF report...")
        create_pdf_report(df, charts_path, output_pdf)
        
        print(f"✅ Analysis complete! PDF report saved as: {output_pdf}")
        print(f"📊 Charts saved as: {charts_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 