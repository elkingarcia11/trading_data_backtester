#!/usr/bin/env python3
"""
Indicator Configuration File
Modify this file to define which indicators you want to use for backtesting
"""

# ============================================================================
# INDICATOR CONFIGURATION
# ============================================================================

# Define which indicators to test and their parameter ranges
INDICATOR_RANGES = {
    # Moving Averages
    'ema': range(9, 10),           # EMA periods to test (currently only 9)
    'vwma': range(17, 18),         # VWMA periods to test (currently only 17)
    'sma': range(10, 21),          # SMA periods to test (10-20)
    
    # Stochastic RSI
    'stoch_rsi_period': range(14, 15),  # Stoch RSI period (currently only 14)
    'stoch_rsi_k': range(3, 4),    # Stoch RSI K period (currently only 3)
    'stoch_rsi_d': range(3, 4),    # Stoch RSI D period (currently only 3)
    
    # RSI
    'rsi': range(14, 15),          # RSI period (currently only 14)
    
    # MACD
    'macd_fast': range(12, 13),    # MACD fast period (currently only 12)
    'macd_slow': range(26, 27),    # MACD slow period (currently only 26)
    'macd_signal': range(9, 10),   # MACD signal period (currently only 9)
    
    # Rate of Change
    'roc': range(3, 21),           # ROC periods to test (3-20)
    'roc_of_roc': range(3, 21),    # ROC of ROC periods (3-20)
    
    # Volatility and Other
    'volatility': range(10, 21),   # Volatility periods (10-20)
    'price_change': range(1, 11),  # Price change periods (1-10)
    'bollinger_period': range(10, 21),  # Bollinger Bands period (10-20)
    'bollinger_std': range(1, 4),  # Bollinger Bands std dev (1-3)
    'atr': range(10, 21)           # ATR periods (10-20)
}

# ============================================================================
# QUICK CONFIGURATION PRESETS
# ============================================================================

# Preset 1: Quick Test (Fast, few combinations)
QUICK_TEST_CONFIG = {
    'ema': range(9, 10),
    'vwma': range(17, 18),
    'roc': range(3, 5),
    'stoch_rsi_period': range(14, 15),
    'stoch_rsi_k': range(3, 4),
    'stoch_rsi_d': range(3, 4),
}
# Total combinations: ~4

# Preset 2: EMA + VWMA Focus
EMA_VWMA_CONFIG = {
    'ema': range(5, 21),           # Test EMA 5-20
    'vwma': range(10, 31),         # Test VWMA 10-30
}
# Total combinations: ~320

# Preset 3: ROC Focus
ROC_CONFIG = {
    'roc': range(3, 21),           # Test ROC 3-20
    'roc_of_roc': range(7, 18),    # Test ROC of ROC 7-17
}
# Total combinations: ~198

# Preset 4: Stoch RSI Focus
STOCH_RSI_CONFIG = {
    'stoch_rsi_period': range(10, 21),  # Test periods 10-20
    'stoch_rsi_k': range(3, 21),        # Test K 3-20
    'stoch_rsi_d': range(3, 21),        # Test D 3-20
}
# Total combinations: ~3,564

# Preset 5: Comprehensive Test (Many combinations)
COMPREHENSIVE_CONFIG = {
    'ema': range(5, 21),           # Test EMA 5-20
    'vwma': range(10, 31),         # Test VWMA 10-30
    'roc': range(3, 21),           # Test ROC 3-20
    'stoch_rsi_period': range(10, 21),  # Test periods 10-20
    'stoch_rsi_k': range(3, 21),   # Test K 3-20
    'stoch_rsi_d': range(3, 21),   # Test D 3-20
    'rsi': range(10, 21),          # Test RSI 10-20
    'macd_fast': range(8, 21),     # Test MACD fast 8-20
    'macd_slow': range(20, 31),    # Test MACD slow 20-30
    'macd_signal': range(5, 16),   # Test MACD signal 5-15
}
# Total combinations: ~20 million (use with caution!)

# Preset 6: Custom Example - EMA + ROC Focus
CUSTOM_EMA_ROC_CONFIG = {
    'ema': range(9, 16),           # Test EMA 9-15
    'vwma': range(17, 25),         # Test VWMA 17-24
    'roc': range(3, 12),           # Test ROC 3-11
    'stoch_rsi_period': range(14, 15),  # Test period 14
    'stoch_rsi_k': range(3, 4),    # Test K 3
    'stoch_rsi_d': range(3, 4),    # Test D 3
}
# Total combinations: ~1,260

# Preset 7: ROC FOCUS (Fixed - includes all required parameters)
CUSTOM_CONFIG_ROC_ROC_OF_ROC = {
    # Moving Averages
    'ema': range(9, 10),           # EMA periods to test (currently only 9)
    'vwma': range(17, 18),         # VWMA periods to test (currently only 17)
    
    # Stochastic RSI (ALL THREE parameters required)
    'stoch_rsi_period': range(14, 15),  # Stoch RSI period (currently only 14)
    'stoch_rsi_k': range(3, 4),    # Stoch RSI K period (currently only 3)
    'stoch_rsi_d': range(3, 4),    # Stoch RSI D period (currently only 3)
    
    # MACD (ALL THREE parameters required)
    'macd_fast': range(12, 13),    # MACD fast period (currently only 12)
    'macd_slow': range(26, 27),    # MACD slow period (currently only 26)
    'macd_signal': range(9, 10),   # MACD signal period (currently only 9)
    
    # Rate of Change
    'roc': range(3, 21),           # ROC periods to test (3-20)
    'roc_of_roc': range(3, 21),    # ROC of ROC periods (3-20)
}


# Preset 8: ROC FOCUS (Fixed - includes all required parameters)
CUSTOM_CONFIG_ROC = {
    # Moving Averages
    'ema': range(9, 10),           # EMA periods to test (currently only 9)
    'vwma': range(17, 18),         # VWMA periods to test (currently only 17)
    
    # Stochastic RSI (ALL THREE parameters required)
    'stoch_rsi_period': range(14, 15),  # Stoch RSI period (currently only 14)
    'stoch_rsi_k': range(3, 4),    # Stoch RSI K period (currently only 3)
    'stoch_rsi_d': range(3, 4),    # Stoch RSI D period (currently only 3)
    
    # MACD (ALL THREE parameters required)
    'macd_fast': range(12, 13),    # MACD fast period (currently only 12)
    'macd_slow': range(26, 27),    # MACD slow period (currently only 26)
    'macd_signal': range(9, 10),   # MACD signal period (currently only 9)
    
    # Rate of Change
    'roc': range(3, 21),           # ROC periods to test (3-20)
}

# Preset 9: ROC OF ROC FOCUS (Fixed - includes all required parameters)
CUSTOM_CONFIG_ROC_OF_ROC = {
    # Moving Averages
    'ema': range(9, 10),           # EMA periods to test (currently only 9)
    'vwma': range(17, 18),         # VWMA periods to test (currently only 17)
    
    # Stochastic RSI (ALL THREE parameters required)
    'stoch_rsi_period': range(14, 15),  # Stoch RSI period (currently only 14)
    'stoch_rsi_k': range(3, 4),    # Stoch RSI K period (currently only 3)
    'stoch_rsi_d': range(3, 4),    # Stoch RSI D period (currently only 3)
    
    # MACD (ALL THREE parameters required)
    'macd_fast': range(12, 13),    # MACD fast period (currently only 12)
    'macd_slow': range(26, 27),    # MACD slow period (currently only 26)
    'macd_signal': range(9, 10),   # MACD signal period (currently only 9)
    
    # Rate of Change
    'roc_of_roc': range(3, 21),    # ROC of ROC periods (3-20)
}

CUSTOM_CONFIG_EMA_VWMA ={
    # Moving Averages
    'ema_fast': range(7, 13, 5),           # EMA periods to test (currently only 9)
    'ema_slow': range(13, 15),         # VWMA periods to test (currently only 17)
    
    # Stochastic RSI (ALL THREE parameters required)
    'stoch_rsi_period': range(14, 15),  # Stoch RSI period (currently only 14)
    'stoch_rsi_k': range(3, 4),    # Stoch RSI K period (currently only 3)
    'stoch_rsi_d': range(3, 4),    # Stoch RSI D period (currently only 3)
    
    # MACD (ALL THREE parameters required)
    'macd_fast': range(12, 13),    # MACD fast period (currently only 12)
    'macd_slow': range(26, 27),    # MACD slow period (currently only 26)
    'macd_signal': range(9, 10),   # MACD signal period (currently only 9)
    
    # Rate of Change
    'roc': range(11, 12),           # ROC periods   
    'roc_of_roc': range(18, 19),    # ROC of ROC periods
}

CUSTOM_CONFIG = {
    # Moving Averages
    'ema': 10,           # EMA periods to test (currently only 9)
    'vwma': 22,           # EMA periods to test (currently only 9)
    
    # Stochastic RSI (ALL THREE parameters required)
    'stoch_rsi_period': 14,  # Stoch RSI period (currently only 14)
    'stoch_rsi_k': 3,    # Stoch RSI K period (currently only 3)
    'stoch_rsi_d': 3,    # Stoch RSI D period (currently only 3)
    
    # MACD (ALL THREE parameters required)
    'macd_fast': 12,    # MACD fast period (currently only 12)
    'macd_slow': 26,    # MACD slow period (currently only 26)
    'macd_signal': 9,   # MACD signal period (currently only 9)
    
    # Rate of Change - Based on analysis showing period+1 is used
    'roc': 11,           # ROC periods (actual implementation uses 11 = period+1)
    'roc_of_roc': 18,    # ROC of ROC periods
}

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
HOW TO USE THIS CONFIGURATION FILE:

1. QUICK START:
   - Modify INDICATOR_RANGES above to change which indicators to test
   - The system will automatically use these settings

2. USE PRESETS:
   - Replace INDICATOR_RANGES with one of the presets:
     INDICATOR_RANGES = QUICK_TEST_CONFIG
     INDICATOR_RANGES = EMA_VWMA_CONFIG
     INDICATOR_RANGES = ROC_CONFIG
     INDICATOR_RANGES = STOCH_RSI_CONFIG
     INDICATOR_RANGES = COMPREHENSIVE_CONFIG

3. CUSTOM CONFIGURATION:
   - Create your own configuration by copying and modifying a preset
   - Example:
     INDICATOR_RANGES = {
         'ema': range(9, 16),           # Test EMA 9-15
         'vwma': range(17, 25),         # Test VWMA 17-24
         'roc': range(3, 12),           # Test ROC 3-11
     }

4. DISABLE INDICATORS:
   - Comment out or remove unwanted indicators:
     # 'stoch_rsi_period': range(14, 15),  # Disabled
     # 'stoch_rsi_k': range(3, 4),        # Disabled

PERFORMANCE ESTIMATES:
- Quick Test: ~1-2 minutes
- EMA/VWMA Focus: ~5-10 minutes  
- ROC Focus: ~3-5 minutes
- Stoch RSI Focus: ~15-30 minutes
- Comprehensive: ~Hours (use with caution!)

RECOMMENDED WORKFLOW:
1. Start with QUICK_TEST_CONFIG
2. Validate results are working
3. Move to EMA_VWMA_CONFIG or ROC_CONFIG
4. Gradually expand based on performance needs
5. Use COMPREHENSIVE_CONFIG only for final optimization
"""

# ============================================================================
# CURRENT ACTIVE CONFIGURATION
# ============================================================================

# Change this line to use a different preset:
INDICATOR_RANGES = CUSTOM_CONFIG  # Start with quick test
# INDICATOR_RANGES = EMA_VWMA_CONFIG  # Uncomment for EMA/VWMA focus
# INDICATOR_RANGES = ROC_CONFIG       # Uncomment for ROC focus
# INDICATOR_RANGES = STOCH_RSI_CONFIG # Uncomment for Stoch RSI focus
# INDICATOR_RANGES = COMPREHENSIVE_CONFIG  # Uncomment for comprehensive test
# INDICATOR_RANGES = ROC_CONFIG
# INDICATOR_RANGES = STOCH_RSI_CONFIG
# INDICATOR_RANGES = COMPREHENSIVE_CONFIG

# Or keep the default configuration:
# INDICATOR_RANGES = INDICATOR_RANGES  # (current settings above) 