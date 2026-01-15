"""
Aventa HFT Pro 2026 - Advanced Order Flow Analysis
Detect and analyze order flow imbalances, VWAP, and market microstructure
"""

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class OrderFlowBar:
    """Order flow analysis for a single bar"""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float
    sell_volume: float
    vwap: float
    twap: float
    delta: float  # buy_volume - sell_volume
    cumulative_delta: float
    on_balance_volume: float
    obv_ema: float
    imbalance_ratio: float  # buy_volume / (buy_volume + sell_volume)
    signal_strength: float


class OrderFlowAnalyzer:
    """Analyze order flow and market microstructure"""
    
    def __init__(self, symbol: str, config: Optional[Dict] = None):
        """
        Initialize order flow analyzer
        
        Args:
            symbol: Trading symbol
            config: Configuration dictionary
        """
        self.symbol = symbol
        self.config = config or {}
        
        # History buffers
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        self.obv_history = deque(maxlen=1000)
        self.vwap_history = deque(maxlen=1000)
        self.delta_history = deque(maxlen=1000)
        self.cumulative_delta = 0
        
        # VWAP calculation
        self.cumsum_pv = 0  # Cumulative sum of price * volume
        self.cumsum_v = 0   # Cumulative sum of volume
        
    def estimate_buy_sell_volume(self, bar: pd.Series, prev_close: float) -> Tuple[float, float]:
        """
        Estimate buy and sell volume using tick rule
        
        Simple heuristic: if close > prev_close, assume volume is mostly buys
        """
        close_price = bar['close']
        volume = bar['tick_volume']
        
        # Tick rule: positive close -> buy volume, negative close -> sell volume
        if close_price > prev_close:
            buy_volume = volume * 0.7  # 70% buy, 30% sell (typical)
            sell_volume = volume * 0.3
        elif close_price < prev_close:
            buy_volume = volume * 0.3
            sell_volume = volume * 0.7
        else:
            buy_volume = volume * 0.5
            sell_volume = volume * 0.5
        
        return buy_volume, sell_volume
    
    def calculate_vwap(self, price: float, volume: float) -> float:
        """
        Calculate Volume Weighted Average Price
        VWAP = Sum(Price * Volume) / Sum(Volume)
        """
        self.cumsum_pv += price * volume
        self.cumsum_v += volume
        
        if self.cumsum_v == 0:
            return price
        
        return self.cumsum_pv / self.cumsum_v
    
    def calculate_twap(self, prices: List[float], period: int = 20) -> float:
        """
        Calculate Time Weighted Average Price
        Simple average of prices over period
        """
        if len(prices) < period:
            return np.mean(prices) if prices else 0
        
        return np.mean(prices[-period:])
    
    def calculate_on_balance_volume(self, close: float, prev_close: float,
                                    volume: float) -> float:
        """
        Calculate On-Balance Volume (OBV)
        OBV accumulates volume based on price direction
        """
        if close > prev_close:
            obv = volume
        elif close < prev_close:
            obv = -volume
        else:
            obv = 0
        
        return obv
    
    def analyze_bar(self, bar: pd.Series, prev_close: float) -> OrderFlowBar:
        """
        Analyze a single OHLCV bar for order flow characteristics
        """
        time = bar['time'] if hasattr(bar['time'], 'strftime') else datetime.now()
        volume = bar['tick_volume']
        close = bar['close']
        
        # Estimate buy/sell volume
        buy_volume, sell_volume = self.estimate_buy_sell_volume(bar, prev_close)
        
        # Calculate metrics
        delta = buy_volume - sell_volume
        self.cumulative_delta += delta
        
        vwap = self.calculate_vwap(close, volume)
        
        # OBV
        obv = self.calculate_on_balance_volume(close, prev_close, volume)
        
        # OBV EMA
        if len(self.obv_history) == 0:
            obv_ema = obv
        else:
            obv_ema = (2 * obv + self.obv_history[-1] * (20 - 1)) / (20 + 1)
        
        # Imbalance ratio (0-1, 0.5 = balanced)
        total_volume = buy_volume + sell_volume
        imbalance_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
        
        # TWAP
        self.price_history.append(close)
        twap = self.calculate_twap(list(self.price_history), period=20)
        
        # Signal strength based on imbalance
        imbalance_strength = abs(imbalance_ratio - 0.5) * 2  # 0-1 scale
        volume_strength = min(volume / (np.mean(self.volume_history) + 1), 2)  # 0-2 scale
        signal_strength = (imbalance_strength + volume_strength) / 2
        
        # Store history
        self.volume_history.append(volume)
        self.vwap_history.append(vwap)
        self.delta_history.append(delta)
        self.obv_history.append(obv_ema)
        
        return OrderFlowBar(
            time=time,
            open=bar['open'],
            high=bar['high'],
            low=bar['low'],
            close=close,
            volume=volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            vwap=vwap,
            twap=twap,
            delta=delta,
            cumulative_delta=self.cumulative_delta,
            on_balance_volume=obv_ema,
            obv_ema=obv_ema,
            imbalance_ratio=imbalance_ratio,
            signal_strength=signal_strength
        )
    
    def detect_accumulation_phase(self, lookback: int = 20) -> bool:
        """
        Detect if market is in accumulation phase
        - Positive cumulative delta
        - Rising OBV
        - Moderate volume
        """
        if len(self.delta_history) < lookback:
            return False
        
        recent_deltas = list(self.delta_history)[-lookback:]
        avg_delta = np.mean(recent_deltas)
        
        recent_obv = list(self.obv_history)[-lookback:]
        obv_trend = recent_obv[-1] - recent_obv[0]
        
        # Accumulation: positive delta + rising OBV
        return avg_delta > 0 and obv_trend > 0
    
    def detect_distribution_phase(self, lookback: int = 20) -> bool:
        """
        Detect if market is in distribution phase
        - Negative cumulative delta
        - Falling OBV
        - High volume
        """
        if len(self.delta_history) < lookback:
            return False
        
        recent_deltas = list(self.delta_history)[-lookback:]
        avg_delta = np.mean(recent_deltas)
        
        recent_obv = list(self.obv_history)[-lookback:]
        obv_trend = recent_obv[-1] - recent_obv[0]
        
        # Distribution: negative delta + falling OBV
        return avg_delta < 0 and obv_trend < 0
    
    def detect_divergence(self, lookback: int = 30) -> Optional[str]:
        """
        Detect price-OBV divergences (bullish or bearish)
        
        Returns:
            'bullish' if price down but OBV up (buy signal)
            'bearish' if price up but OBV down (sell signal)
            None if no divergence
        """
        if len(self.price_history) < lookback:
            return None
        
        prices = list(self.price_history)[-lookback:]
        obvs = list(self.obv_history)[-lookback:]
        
        price_trend = prices[-1] - prices[0]
        obv_trend = obvs[-1] - obvs[0]
        
        # Bullish divergence: price down, OBV up
        if price_trend < 0 and obv_trend > 0:
            return 'bullish'
        
        # Bearish divergence: price up, OBV down
        elif price_trend > 0 and obv_trend < 0:
            return 'bearish'
        
        return None
    
    def detect_volume_surge(self, threshold_multiplier: float = 1.5) -> bool:
        """
        Detect unusual volume surge
        
        Returns True if current volume > threshold_multiplier * average volume
        """
        if len(self.volume_history) < 20:
            return False
        
        avg_volume = np.mean(list(self.volume_history)[-20:])
        current_volume = self.volume_history[-1] if self.volume_history else 0
        
        return current_volume > avg_volume * threshold_multiplier
    
    def get_imbalance_strength(self, lookback: int = 10) -> float:
        """
        Calculate imbalance strength (how strong is the current buy/sell pressure)
        
        Returns:
            Value 0-2: 
            - 1.0 = balanced
            - 0-1.0 = sell pressure
            - 1.0-2.0 = buy pressure
        """
        if not self.delta_history:
            return 1.0
        
        recent_deltas = list(self.delta_history)[-lookback:]
        avg_delta = np.mean(recent_deltas)
        max_delta = np.abs(max(recent_deltas, key=abs)) if recent_deltas else 1
        
        if max_delta == 0:
            return 1.0
        
        return 1.0 + (avg_delta / max_delta)


class AdvancedOrderFlowSignal:
    """Generate trading signals from order flow analysis"""
    
    def __init__(self, symbol: str, config: Optional[Dict] = None):
        """Initialize order flow signal generator"""
        self.symbol = symbol
        self.config = config or {}
        self.analyzer = OrderFlowAnalyzer(symbol, config)
    
    def generate_signal(self, bars: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate trading signal from order flow analysis
        
        Args:
            bars: DataFrame with OHLCV data
            
        Returns:
            (signal, confidence)
            signal: 1 for BUY, -1 for SELL, 0 for NO SIGNAL
            confidence: 0-1 confidence level
        """
        if len(bars) < 3:
            return 0, 0.0
        
        # Analyze current and previous bars
        current_bar = bars.iloc[-1]
        prev_close = bars.iloc[-2]['close']
        
        flow = self.analyzer.analyze_bar(current_bar, prev_close)
        
        confidence = 0.0
        signal = 0
        
        # Signal 1: Accumulation with volume surge
        if self.analyzer.detect_accumulation_phase():
            if self.analyzer.detect_volume_surge(threshold_multiplier=1.3):
                signal = 1  # BUY
                confidence = min(0.9, flow.signal_strength * 0.8)
        
        # Signal 2: Distribution with volume surge
        if self.analyzer.detect_distribution_phase():
            if self.analyzer.detect_volume_surge(threshold_multiplier=1.3):
                signal = -1  # SELL
                confidence = min(0.9, flow.signal_strength * 0.8)
        
        # Signal 3: Bullish divergence
        if self.analyzer.detect_divergence() == 'bullish':
            signal = 1
            confidence = min(0.8, 0.6 + flow.imbalance_ratio * 0.2)
        
        # Signal 4: Bearish divergence
        if self.analyzer.detect_divergence() == 'bearish':
            signal = -1
            confidence = min(0.8, 0.6 + (1 - flow.imbalance_ratio) * 0.2)
        
        # Signal 5: Strong imbalance
        imbalance_strength = self.analyzer.get_imbalance_strength()
        if imbalance_strength > 1.5:  # Strong buy pressure
            signal = 1
            confidence = min(0.7, (imbalance_strength - 1.0) * 0.35)
        elif imbalance_strength < 0.5:  # Strong sell pressure
            signal = -1
            confidence = min(0.7, (1.0 - imbalance_strength) * 0.35)
        
        return signal, confidence


class VolumeProfileAnalyzer:
    """Analyze volume profile and price levels"""
    
    def __init__(self, bin_size: float = 0.0001):  # 1 pip bins
        """
        Initialize volume profile analyzer
        
        Args:
            bin_size: Size of price bins for volume profile
        """
        self.bin_size = bin_size
        self.volume_by_price = {}
    
    def add_bar(self, high: float, low: float, volume: float):
        """
        Add bar to volume profile
        
        Args:
            high: Bar high
            low: Bar low
            volume: Bar volume
        """
        # Distribute volume across price range
        start_bin = int(low / self.bin_size)
        end_bin = int(high / self.bin_size)
        
        for bin_num in range(start_bin, end_bin + 1):
            price_level = bin_num * self.bin_size
            if price_level not in self.volume_by_price:
                self.volume_by_price[price_level] = 0
            self.volume_by_price[price_level] += volume / (end_bin - start_bin + 1)
    
    def get_value_area(self, percentage: float = 0.70) -> Tuple[float, float]:
        """
        Get value area (price range containing X% of volume)
        
        Args:
            percentage: Percentage of volume to include (default 70%)
            
        Returns:
            (low_price, high_price) of value area
        """
        if not self.volume_by_price:
            return None, None
        
        sorted_levels = sorted(self.volume_by_price.items(), key=lambda x: x[1], reverse=True)
        target_volume = sum(v for p, v in self.volume_by_price.items()) * percentage
        
        cumulative_volume = 0
        included_levels = []
        
        for price, volume in sorted_levels:
            included_levels.append(price)
            cumulative_volume += volume
            if cumulative_volume >= target_volume:
                break
        
        if not included_levels:
            return None, None
        
        return min(included_levels), max(included_levels)
    
    def get_point_of_control(self) -> float:
        """Get price level with highest volume (point of control)"""
        if not self.volume_by_price:
            return None
        
        poc = max(self.volume_by_price.items(), key=lambda x: x[1])[0]
        return poc
    
    def get_resistance_support(self, top_n: int = 3) -> Tuple[List[float], List[float]]:
        """
        Get top resistance and support levels
        
        Args:
            top_n: Number of levels to return
            
        Returns:
            (resistance_levels, support_levels)
        """
        if not self.volume_by_price:
            return [], []
        
        sorted_levels = sorted(self.volume_by_price.items(), key=lambda x: x[1], reverse=True)
        
        resistance = [p for p, v in sorted_levels[:top_n]]
        support = [p for p, v in sorted_levels[:top_n]]
        
        return sorted(resistance, reverse=True), sorted(support)
