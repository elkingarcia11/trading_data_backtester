#!/usr/bin/env python3
"""
Generate all possible signal combinations for indicators
"""

import itertools
from typing import List, Dict

def generate_all_signal_combos_with_ignored() -> List[Dict]:
    """
    Generate all possible MACD signal combinations with fixed conditions for other indicators.
    Only MACD components (histogram, signal, line) will vary.
    Other indicators (roc, stoch_rsi_k, ema vs vwma) are constant per buy/sell type.
    
    Returns:
        List[Dict]: List of dictionaries containing signal combinations
    """
    
    # Define only MACD indicators that will vary (including empty for ignored)
    indicators = {
        'macd_histogram': ['>', '<', '<=', '>=', ''],
        'macd_signal': ['>', '<', '<=', '>=', ''],
        'macd_line': ['>', '<', '<=', '>=', ''],
    }
    
    # Create all possible combinations
    all_combinations = []
    
    # Generate all condition combinations for all indicators
    condition_lists = list(indicators.values())
    for condition_combo in itertools.product(*condition_lists):
        combo_dict = {}
        for i, indicator in enumerate(indicators.keys()):
            # Only add indicators that have non-empty conditions
            if condition_combo[i] != '':
                combo_dict[indicator] = condition_combo[i]
        
        # Add all combinations, even if all MACD indicators are ignored
        # (empty combo means relying only on constant conditions)
        all_combinations.append(combo_dict)
    
    return all_combinations

def is_valid_buy_combo(combo: Dict) -> bool:
    """
    Check if a combination is valid for BUY signals.
    Buy criteria: roc > 0 AND stoch_rsi_k > d AND ema > vwma (all constant)
    Only MACD components can vary.
    
    Args:
        combo (Dict): Signal combination (only contains MACD components)
        
    Returns:
        bool: True if valid buy combo (always True since all MACD combos are valid for buy)
    """
    return True

def is_valid_sell_combo(combo: Dict) -> bool:
    """
    Check if a combination is valid for SELL signals.
    Sell criteria: roc < 0 AND stoch_rsi_k < d AND ema < vwma (all constant)
    Only MACD components can vary.
    
    Args:
        combo (Dict): Signal combination (only contains MACD components)
        
    Returns:
        bool: True if valid sell combo (always True since all MACD combos are valid for sell)
    """
    return True

def generate_filtered_buy_sell_combos() -> tuple[List[Dict], List[Dict]]:
    """
    Generate filtered buy and sell combinations with constant conditions and varying MACD.
    
    Buy criteria: roc > 0 AND stoch_rsi_k > d AND ema > vwma (constants) + MACD variations
    Sell criteria: roc < 0 AND stoch_rsi_k < d AND ema < vwma (constants) + MACD variations
    
    Returns:
        tuple: (buy_combinations, sell_combinations)
    """
    # Generate all possible MACD combinations first
    all_combinations = generate_all_signal_combos_with_ignored()
    
    # Create buy combinations with constant conditions
    buy_combinations = []
    for combo in all_combinations:
        if is_valid_buy_combo(combo):
            buy_combo = combo.copy()  # Create a copy to avoid modifying original
            # Add constant buy conditions
            buy_combo['roc'] = '>'  # roc > 0 (constant for buy)
            buy_combo['stoch_rsi_k'] = '>'  # stoch k > d (constant for buy)
            buy_combo['ema'] = '>'  # ema > vwma (constant for buy)
            buy_combinations.append(buy_combo)
    
    # Create sell combinations with constant conditions
    sell_combinations = []
    for combo in all_combinations:
        if is_valid_sell_combo(combo):
            sell_combo = combo.copy()  # Create a copy to avoid modifying original
            # Add constant sell conditions
            sell_combo['roc'] = '<'  # roc < 0 (constant for sell)
            sell_combo['stoch_rsi_k'] = '<'  # stoch rsi k < d (constant for sell)
            sell_combo['ema'] = '<'  # ema < vwma (constant for sell)
            sell_combinations.append(sell_combo)
    
    return buy_combinations, sell_combinations



def save_combinations_to_file(combinations: List[Dict], filename: str):
    """
    Save combinations to a file for later use.
    
    Args:
        combinations (List[Dict]): List of signal combinations
        filename (str): Output filename
    """
    import json
    
    with open(filename, 'w') as f:
        json.dump(combinations, f, indent=2)
    
    print(f"Saved {len(combinations)} combinations to {filename}")

def load_combinations_from_file(filename: str) -> List[Dict]:
    """
    Load combinations from a file.
    
    Args:
        filename (str): Input filename
        
    Returns:
        List[Dict]: List of signal combinations
    """
    import json
    
    with open(filename, 'r') as f:
        combinations = json.load(f)
    
    return combinations

if __name__ == "__main__":
    print("Generating MACD signal combinations...")
    #all_combos_with_ignored = generate_all_signal_combos_with_ignored()
    #print(f"All MACD combinations (with ignored): {len(all_combos_with_ignored)}")
    
    print("\nGenerating buy/sell combinations with constant conditions...")
    buy_combinations, sell_combinations = generate_filtered_buy_sell_combos()
    
    print(f"Buy combinations (roc > 0, stoch_rsi_k > d, ema > vwma + MACD variations): {len(buy_combinations)}")
    print(f"Sell combinations (roc < 0, stoch_rsi_k < d, ema < vwma + MACD variations): {len(sell_combinations)}")
    
    # Save all combinations
    #save_combinations_to_file(all_combos_with_ignored, 'signal_combinations_all_with_ignored.json')
    save_combinations_to_file(buy_combinations, 'signal_combinations_buy_filtered.json')
    save_combinations_to_file(sell_combinations, 'signal_combinations_sell_filtered.json')
    
    print(f"\nFiles saved:")
    #print(f"  - signal_combinations_all_with_ignored.json ({len(all_combos_with_ignored)} MACD combinations)")
    print(f"  - signal_combinations_buy_filtered.json ({len(buy_combinations)} buy combinations)")
    print(f"  - signal_combinations_sell_filtered.json ({len(sell_combinations)} sell combinations)")
    
    print(f"\nTotal combinations to test: {len(buy_combinations) * len(sell_combinations):,}")
    print("This is the cartesian product that will be tested if TEST_ALL_COMBINATIONS=True")
    print("\nNote: All combinations now focus on MACD variations with constant conditions:")
    print("  Buy: roc > 0, stoch_rsi_k > d, ema > vwma")
    print("  Sell: roc < 0, stoch_rsi_k < d, ema < vwma")