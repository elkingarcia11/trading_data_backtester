#!/usr/bin/env python3
"""
Test script to validate the indicator calculator by comparing calculated indicators
with the actual pre-calculated data in the options_actual folder.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the indicator-calculator directory to the path
sys.path.append('indicator-calculator')
from indicator_calculator import IndicatorCalculator

def load_csv_with_fallback_encoding(filepath):
    """Load CSV with fallback encoding handling"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            break
    
    return None

def get_default_indicator_periods():
    """Get periods from CUSTOM_CONFIG"""
    # Import the config
    import indicator_config
    
    # Convert ranges to single values where needed and ensure all required keys exist
    config = {}
    
    # Handle range or single values from CUSTOM_CONFIG
    for key, value in indicator_config.CUSTOM_CONFIG.items():
        if hasattr(value, '__iter__') and not isinstance(value, (str, int)):
            # It's a range, take the first value
            config[key] = list(value)[0]
        else:
            # It's a single value
            config[key] = value
    
    return config

def calculate_indicators_for_file(filepath, calculator, indicator_periods):
    """Calculate all indicators for a single CSV file"""
    print(f"Processing: {os.path.basename(filepath)}")
    
    # Load the data
    df = load_csv_with_fallback_encoding(filepath)
    if df is None:
        print(f"  ERROR: Could not load {filepath}")
        return None
    
    # Ensure we have the required columns
    required_columns = ['last_price', 'timestamp']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"  ERROR: Missing required columns: {missing_columns}")
        return None
    
    # Sort by timestamp to ensure proper order
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    try:
        # Calculate all indicators using the appropriate price column for options
        result_df = calculator.calculate_all_indicators(
            df, 
            indicator_periods=indicator_periods,
            price_column='last_price'  # Use last_price for options data
        )
        
        print(f"  SUCCESS: Calculated indicators for {len(result_df)} rows")
        return result_df
        
    except Exception as e:
        print(f"  ERROR: Failed to calculate indicators: {e}")
        return None

def compare_indicators(calculated_df, actual_df, filename):
    """Compare calculated indicators with actual pre-calculated ones"""
    comparison_results = {
        'filename': filename,
        'total_rows': len(calculated_df),
        'indicators_compared': [],
        'matches': {},
        'differences': {},
        'missing_in_actual': [],
        'missing_in_calculated': []
    }
    
    # Find common indicator columns
    calculated_indicators = [col for col in calculated_df.columns 
                           if col in ['ema', 'vwma', 'roc', 'roc_of_roc', 'macd_line', 'macd_signal', 'stoch_rsi_k', 'stoch_rsi_d']]
    
    actual_indicators = [col for col in actual_df.columns 
                        if col in ['ema', 'vwma', 'roc', 'roc_of_roc', 'macd_line', 'macd_signal', 'stoch_rsi_k', 'stoch_rsi_d']]
    
    # Find missing indicators
    comparison_results['missing_in_actual'] = [col for col in calculated_indicators if col not in actual_indicators]
    comparison_results['missing_in_calculated'] = [col for col in actual_indicators if col not in calculated_indicators]
    
    # Compare common indicators
    common_indicators = [col for col in calculated_indicators if col in actual_indicators]
    comparison_results['indicators_compared'] = common_indicators
    
    for indicator in common_indicators:
        calc_values = calculated_df[indicator].dropna()
        actual_values = actual_df[indicator].dropna()
        
        if len(calc_values) == 0 and len(actual_values) == 0:
            comparison_results['matches'][indicator] = 'Both empty'
            continue
            
        if len(calc_values) == 0:
            comparison_results['differences'][indicator] = 'Calculated values are all NaN'
            continue
            
        if len(actual_values) == 0:
            comparison_results['differences'][indicator] = 'Actual values are all NaN'
            continue
        
        # Align the data by index for comparison
        min_len = min(len(calc_values), len(actual_values))
        if min_len == 0:
            comparison_results['differences'][indicator] = 'No overlapping data'
            continue
            
        calc_subset = calc_values.iloc[-min_len:].reset_index(drop=True)
        actual_subset = actual_values.iloc[-min_len:].reset_index(drop=True)
        
        # Calculate differences
        diff = np.abs(calc_subset - actual_subset)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        # Consider values matching if they're within a small tolerance
        tolerance = 1e-6
        matching_ratio = np.sum(diff < tolerance) / len(diff)
        
        if matching_ratio > 0.95:  # 95% of values match within tolerance
            comparison_results['matches'][indicator] = {
                'status': 'MATCH',
                'matching_ratio': matching_ratio,
                'max_diff': max_diff,
                'mean_diff': mean_diff
            }
        else:
            comparison_results['differences'][indicator] = {
                'status': 'DIFFERENT',
                'matching_ratio': matching_ratio,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'sample_calc': calc_subset.head(5).tolist(),
                'sample_actual': actual_subset.head(5).tolist()
            }
    
    return comparison_results

def save_comparison_data(calculated_df, actual_df, filename):
    """Save calculated vs actual data for detailed examination"""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path('indicator_comparison_output')
        output_dir.mkdir(exist_ok=True)
        
        # Get indicator columns that we calculated
        calculated_indicators = [col for col in calculated_df.columns 
                               if col in ['ema', 'vwma', 'roc', 'roc_of_roc', 'macd_line', 'macd_signal', 'stoch_rsi_k', 'stoch_rsi_d']]
        
        actual_indicators = [col for col in actual_df.columns 
                           if col in ['ema', 'vwma', 'roc', 'roc_of_roc', 'macd_line', 'macd_signal', 'stoch_rsi_k', 'stoch_rsi_d']]
        
        # Common indicators
        common_indicators = [col for col in calculated_indicators if col in actual_indicators]
        
        if not common_indicators:
            print(f"  No common indicators to save for {filename}")
            return
        
        # Create comparison dataframe
        comparison_data = {
            'timestamp': calculated_df['timestamp'],
            'last_price': calculated_df['last_price']
        }
        
        # Add calculated vs actual for each indicator
        for indicator in common_indicators:
            comparison_data[f'{indicator}_calculated'] = calculated_df[indicator]
            comparison_data[f'{indicator}_actual'] = actual_df[indicator]
            
            # Calculate difference
            diff = calculated_df[indicator] - actual_df[indicator]
            comparison_data[f'{indicator}_difference'] = diff
            
            # Calculate percentage difference where applicable
            actual_nonzero = actual_df[indicator].replace(0, np.nan)
            pct_diff = (diff / actual_nonzero) * 100
            comparison_data[f'{indicator}_pct_diff'] = pct_diff
        
        # Create DataFrame and save
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        output_file = output_dir / f'{filename.replace(".csv", "_comparison.csv")}'
        comparison_df.to_csv(output_file, index=False)
        
        print(f"  SAVED: Comparison data to {output_file}")
        
        # Also save a summary file with key statistics
        summary_file = output_dir / f'{filename.replace(".csv", "_summary.txt")}'
        with open(summary_file, 'w') as f:
            f.write(f"INDICATOR COMPARISON SUMMARY for {filename}\n")
            f.write("=" * 60 + "\n\n")
            
            for indicator in common_indicators:
                calc_col = f'{indicator}_calculated'
                actual_col = f'{indicator}_actual'
                diff_col = f'{indicator}_difference'
                
                if calc_col in comparison_df.columns and actual_col in comparison_df.columns:
                    calc_data = comparison_df[calc_col].dropna()
                    actual_data = comparison_df[actual_col].dropna()
                    diff_data = comparison_df[diff_col].dropna()
                    
                    if len(calc_data) > 0 and len(actual_data) > 0:
                        f.write(f"{indicator.upper()}:\n")
                        f.write(f"  Calculated values: {len(calc_data)} non-null\n")
                        f.write(f"  Actual values: {len(actual_data)} non-null\n")
                        f.write(f"  Max difference: {diff_data.abs().max():.6f}\n")
                        f.write(f"  Mean difference: {diff_data.mean():.6f}\n")
                        f.write(f"  Std difference: {diff_data.std():.6f}\n")
                        
                        # First valid indices
                        calc_first = comparison_df[calc_col].first_valid_index()
                        actual_first = comparison_df[actual_col].first_valid_index()
                        f.write(f"  First valid calc index: {calc_first}\n")
                        f.write(f"  First valid actual index: {actual_first}\n")
                        
                        # Sample values
                        if len(calc_data) >= 5:
                            f.write(f"  First 5 calculated: {calc_data.head(5).tolist()}\n")
                            f.write(f"  First 5 actual: {actual_data.head(5).tolist()}\n")
                        
                        f.write("\n")
        
        print(f"  SAVED: Summary to {summary_file}")
        
    except Exception as e:
        print(f"  ERROR: Could not save comparison data for {filename}: {e}")

def generate_comparison_report(all_results):
    """Generate a comprehensive comparison report"""
    print("\n" + "="*80)
    print("INDICATOR CALCULATOR VALIDATION REPORT")
    print("="*80)
    
    total_files = len(all_results)
    successful_files = [r for r in all_results if r is not None]
    
    print(f"\nOVERVIEW:")
    print(f"  Total files processed: {total_files}")
    print(f"  Successful calculations: {len(successful_files)}")
    print(f"  Failed calculations: {total_files - len(successful_files)}")
    
    if not successful_files:
        print("\nNo successful calculations to compare!")
        return
    
    print(f"\nDETAILED RESULTS:")
    print("-" * 80)
    
    overall_matches = {}
    overall_differences = {}
    
    for result in successful_files:
        print(f"\nFile: {result['filename']}")
        print(f"  Rows processed: {result['total_rows']}")
        print(f"  Indicators compared: {len(result['indicators_compared'])}")
        
        if result['missing_in_calculated']:
            print(f"  Missing in calculated: {result['missing_in_calculated']}")
        
        if result['missing_in_actual']:
            print(f"  Missing in actual: {result['missing_in_actual']}")
        
        # Matches
        if result['matches']:
            print(f"  MATCHES ({len(result['matches'])}):")
            for indicator, match_info in result['matches'].items():
                if isinstance(match_info, dict):
                    print(f"    {indicator}: {match_info['status']} (ratio: {match_info['matching_ratio']:.3f}, max_diff: {match_info['max_diff']:.6f})")
                else:
                    print(f"    {indicator}: {match_info}")
                
                # Track overall results
                if indicator not in overall_matches:
                    overall_matches[indicator] = 0
                overall_matches[indicator] += 1
        
        # Differences
        if result['differences']:
            print(f"  DIFFERENCES ({len(result['differences'])}):")
            for indicator, diff_info in result['differences'].items():
                if isinstance(diff_info, dict):
                    print(f"    {indicator}: {diff_info['status']} (ratio: {diff_info['matching_ratio']:.3f}, max_diff: {diff_info['max_diff']:.6f})")
                    print(f"      Sample calculated: {diff_info['sample_calc']}")
                    print(f"      Sample actual:     {diff_info['sample_actual']}")
                else:
                    print(f"    {indicator}: {diff_info}")
                
                # Track overall results
                if indicator not in overall_differences:
                    overall_differences[indicator] = 0
                overall_differences[indicator] += 1
    
    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY ACROSS ALL FILES")
    print("="*80)
    
    all_indicators = set(overall_matches.keys()) | set(overall_differences.keys())
    
    for indicator in sorted(all_indicators):
        matches = overall_matches.get(indicator, 0)
        differences = overall_differences.get(indicator, 0)
        total = matches + differences
        
        if total > 0:
            success_rate = matches / total * 100
            print(f"{indicator:15} - Success: {matches}/{total} ({success_rate:.1f}%)")
    
    print(f"\nOVERALL SUCCESS RATE:")
    total_comparisons = sum(overall_matches.values()) + sum(overall_differences.values())
    total_successes = sum(overall_matches.values())
    
    if total_comparisons > 0:
        overall_success_rate = total_successes / total_comparisons * 100
        print(f"  {total_successes}/{total_comparisons} ({overall_success_rate:.1f}%) indicators matched across all files")

def main():
    """Main test function"""
    print("Starting Indicator Calculator Validation Test")
    print("=" * 50)
    
    # Initialize calculator
    calculator = IndicatorCalculator()
    indicator_periods = get_default_indicator_periods()
    
    print(f"Using indicator configuration: {indicator_periods}")
    print(f"Source: CUSTOM_CONFIG from indicator_config.py")
    print()
    
    # Get file lists
    options_dir = Path('data/options')
    options_actual_dir = Path('data/options_actual')
    
    options_files = list(options_dir.glob('*.csv'))
    options_actual_files = list(options_actual_dir.glob('*.csv'))
    
    print(f"Found {len(options_files)} files in data/options")
    print(f"Found {len(options_actual_files)} files in data/options_actual")
    
    # Find matching files
    options_basenames = {f.name for f in options_files}
    actual_basenames = {f.name for f in options_actual_files}
    common_files = options_basenames & actual_basenames
    
    print(f"Found {len(common_files)} common files to test")
    
    if not common_files:
        print("No common files found! Cannot proceed with comparison.")
        return
    
    all_results = []
    
    for filename in sorted(common_files):
        print(f"\n{'-'*60}")
        print(f"Testing: {filename}")
        
        options_file = options_dir / filename
        actual_file = options_actual_dir / filename
        
        # Calculate indicators on the options file
        calculated_df = calculate_indicators_for_file(
            str(options_file), 
            calculator, 
            indicator_periods
        )
        
        if calculated_df is None:
            print(f"  Skipping {filename} due to calculation failure")
            all_results.append(None)
            continue
        
        # Load the actual file for comparison
        actual_df = load_csv_with_fallback_encoding(str(actual_file))
        if actual_df is None:
            print(f"  ERROR: Could not load actual file {actual_file}")
            all_results.append(None)
            continue
        
        # Compare the results
        comparison_result = compare_indicators(calculated_df, actual_df, filename)
        all_results.append(comparison_result)
        
        # Save comparison data for examination
        save_comparison_data(calculated_df, actual_df, filename)
    
    # Generate final report
    generate_comparison_report(all_results)

if __name__ == "__main__":
    main()