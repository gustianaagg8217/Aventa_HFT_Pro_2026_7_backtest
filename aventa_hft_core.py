"""
Aventa HFT Pro 2026 - Ultra Low Latency Trading Engine
Advanced High-Frequency Trading System for MetaTrader 5
Created: December 2025
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import threading
import logging
from queue import Queue, PriorityQueue
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Ultra-fast tick data structure"""
    timestamp: float
    bid: float
    ask: float
    last: float
    volume: int
    spread: float
    
    def __post_init__(self):
        self.mid_price = (self.bid + self.ask) / 2


@dataclass
class OrderFlowData:
    """Order flow analysis data"""
    timestamp: float
    buy_volume: float
    sell_volume: float
    delta: float
    cumulative_delta: float
    imbalance_ratio: float


@dataclass
class Signal:
    """Trading signal with priority"""
    timestamp: float
    signal_type: str  # 'BUY', 'SELL', 'CLOSE'
    strength: float
    price: float
    stop_loss: float
    take_profit: float
    volume: float
    reason: str
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp


class UltraLowLatencyEngine:
    """Core HFT engine with microsecond precision"""
    
    def __init__(self, symbol: str, config: Dict, risk_manager=None):
        self.symbol = symbol
        self.config = config
        self.risk_manager = risk_manager
        
        # Performance optimization
        self.tick_buffer = deque(maxlen=10000)
        self.orderflow_buffer = deque(maxlen=5000)
        self.signal_queue = PriorityQueue(maxsize=1000)
        
        # Market data
        self.last_tick: Optional[TickData] = None
        self.last_bid = 0.0
        self.last_ask = 0.0
        
        # Symbol info (will be set during initialization)
        self.symbol_point = 0.0
        self.stops_level = 0
        
        # Order flow tracking
        self.cumulative_delta = 0.0
        self.volume_profile = {}
        
        # Performance metrics
        self.latency_samples = deque(maxlen=1000)
        self.execution_times = deque(maxlen=1000)
        
        # State
        self.is_running = False
        self.position_type = None  # None, 'BUY', 'SELL'
        self.position_volume = 0.0
        self.position_price = 0.0
        
        # Threading
        self.data_thread = None
        self.execution_thread = None
        self.analysis_thread = None
    
    @staticmethod
    def get_filling_mode(mode_str: str):
        """Convert filling mode string to MT5 constant"""
        mode_map = {
            'FOK': mt5.ORDER_FILLING_FOK,    # Fill or Kill
            'IOC': mt5.ORDER_FILLING_IOC,    # Immediate or Cancel
            'RETURN': mt5.ORDER_FILLING_RETURN  # Return/Market
        }
        return mode_map.get(mode_str.upper(), mt5.ORDER_FILLING_FOK)
        
    def initialize(self) -> bool:
        """Initialize MT5 connection"""
        try:
            # Get MT5 path from config, use default if not provided
            mt5_path = self.config.get('mt5_path', 'C:\\Program Files\\XM Global MT5\\terminal64.exe')
            
            if not mt5.initialize(mt5_path):
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                logger.error(f"MT5 path: {mt5_path}")
                logger.error(f"  -> Check if MT5 is installed at this location")
                logger.error(f"  -> Update MT5 path in GUI settings if needed")
                return False
            
            # Check symbol
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found")
                return False
            
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(f"Failed to select {self.symbol}")
                    return False
            
            logger.info(f"✓ MT5 initialized successfully for {self.symbol}")
            logger.info(f"  MT5 Path: {mt5_path}")
            logger.info(f"  Spread: {symbol_info.spread} points")
            logger.info(f"  Tick size: {symbol_info.trade_tick_size}")
            logger.info(f"  Tick value: {symbol_info.trade_tick_value}")
            logger.info(f"  Point: {symbol_info.point}")
            logger.info(f"  Stops level: {symbol_info.trade_stops_level} points")
            logger.info(f"  Filling mode: {self.config.get('filling_mode', 'FOK')}")
            logger.info(f"  SL Multiplier: {self.config.get('sl_multiplier', 2.0)}x ATR")
            
            # TP Mode logging
            tp_mode = self.config.get('tp_mode', 'RiskReward')
            if tp_mode == 'FixedDollar':
                tp_amount = self.config.get('tp_dollar_amount', 0.5)
                logger.info(f"  TP Mode: Fixed Dollar (${tp_amount:.2f} per position)")
            else:
                logger.info(f"  TP Mode: Risk:Reward (1:{self.config.get('risk_reward_ratio', 2.0)})")
            
            # Store symbol info
            self.symbol_point = symbol_info.point
            self.stops_level = symbol_info.trade_stops_level
            
            # Calculate actual spread in price terms
            spread_price = symbol_info.spread * symbol_info.point
            logger.info(f"  Spread (price): {spread_price:.5f}")
            
            # Show configured max spread
            max_spread = self.config.get('max_spread', 2.0)
            logger.info(f"  Max Spread Setting: {max_spread:.5f}")
            
            # Calculate minimum SL/TP distance based on stops level
            min_distance = self.stops_level * self.symbol_point
            logger.info(f"  Min SL/TP distance: {min_distance:.5f} ({self.stops_level} points)")
            
            if spread_price > max_spread:
                logger.warning(f"  ⚠️ Current spread ({spread_price:.5f}) > Max allowed ({max_spread:.5f})")
                logger.warning(f"     Bot may not trade until spread narrows!")
                logger.warning(f"     Consider increasing 'Max Spread' setting")
            else:
                logger.info(f"  ✓ Spread OK for trading")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    def get_tick_ultra_fast(self) -> Optional[TickData]:
        """Ultra-fast tick retrieval with microsecond timestamps"""
        start_time = time.perf_counter()
        
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return None
            
            tick_data = TickData(
                timestamp=tick.time + tick.time_msc / 1000.0,
                bid=tick.bid,
                ask=tick.ask,
                last=tick.last,
                volume=tick.volume,
                spread=(tick.ask - tick.bid)
            )
            
            # Track latency
            latency = (time.perf_counter() - start_time) * 1000000  # microseconds
            self.latency_samples.append(latency)
            
            return tick_data
            
        except Exception as e:
            logger.error(f"Tick retrieval error: {e}")
            return None
    
    def calculate_order_flow(self, tick: TickData) -> OrderFlowData:
        """Advanced order flow analysis"""
        if self.last_tick is None:
            self.last_tick = tick
            return None
        
        # Determine aggressor
        price_change = tick.last - self.last_tick.last
        volume_delta = 0.0
        
        if price_change > 0:
            # Buy aggressor
            buy_volume = tick.volume
            sell_volume = 0
            volume_delta = tick.volume
        elif price_change < 0:
            # Sell aggressor
            buy_volume = 0
            sell_volume = tick.volume
            volume_delta = -tick.volume
        else:
            # Use bid/ask to determine
            if tick.last >= tick.mid_price:
                buy_volume = tick.volume
                sell_volume = 0
                volume_delta = tick.volume
            else:
                buy_volume = 0
                sell_volume = tick.volume
                volume_delta = -tick.volume
        
        self.cumulative_delta += volume_delta
        
        # Calculate imbalance
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            imbalance_ratio = (buy_volume - sell_volume) / total_volume
        else:
            imbalance_ratio = 0.0
        
        orderflow = OrderFlowData(
            timestamp=tick.timestamp,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            delta=volume_delta,
            cumulative_delta=self.cumulative_delta,
            imbalance_ratio=imbalance_ratio
        )
        
        self.last_tick = tick
        return orderflow
    
    def analyze_microstructure(self) -> Dict:
        """Analyze market microstructure for HFT opportunities"""
        if len(self.tick_buffer) < 100:
            return {}
        
        recent_ticks = list(self.tick_buffer)[-100:]
        
        # Spread analysis
        spreads = [t.spread for t in recent_ticks]
        avg_spread = np.mean(spreads)
        spread_volatility = np.std(spreads)
        
        # Price momentum (ultra-short term)
        prices = [t.mid_price for t in recent_ticks]
        price_change = prices[-1] - prices[0]
        price_velocity = price_change / len(prices)
        
        # Order flow imbalance
        if len(self.orderflow_buffer) > 0:
            recent_flow = list(self.orderflow_buffer)[-50:]
            avg_delta = np.mean([f.delta for f in recent_flow])
            cumul_delta = recent_flow[-1].cumulative_delta if recent_flow else 0
        else:
            avg_delta = 0
            cumul_delta = 0
        
        # Volatility estimation
        returns = np.diff([t.mid_price for t in recent_ticks])
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        return {
            'avg_spread': avg_spread,
            'spread_volatility': spread_volatility,
            'price_velocity': price_velocity,
            'price_change': price_change,
            'avg_delta': avg_delta,
            'cumulative_delta': cumul_delta,
            'volatility': volatility,
            'tick_count': len(recent_ticks)
        }
    
    def generate_signal(self, microstructure: Dict) -> Optional[Signal]:
        """Generate trading signal based on microstructure analysis"""
        if not microstructure:
            return None
        
        current_tick = self.last_tick
        if current_tick is None:
            return None
        
        # Signal generation parameters
        min_delta_threshold = self.config.get('min_delta_threshold', 100)
        min_velocity_threshold = self.config.get('min_velocity_threshold', 0.00001)
        spread_threshold = self.config.get('max_spread', 0.0001)
        
        # Check spread condition with logging
        if microstructure['avg_spread'] > spread_threshold:
            if np.random.random() < 0.05:  # Log 5% of spread rejections
                logger.warning(f"⚠️ SPREAD REJECT: {microstructure['avg_spread']:.5f} > {spread_threshold:.5f} (max)")
                logger.warning(f"   → Increase 'Max Spread' setting to at least {microstructure['avg_spread']*1.5:.5f}")
            return None
        
        signal_strength = 0.0
        signal_type = None
        reason = []
        
        # Order flow signal
        if microstructure['cumulative_delta'] > min_delta_threshold:
            signal_strength += 0.4
            signal_type = 'BUY'
            reason.append(f"Positive delta: {microstructure['cumulative_delta']:.0f}")
        elif microstructure['cumulative_delta'] < -min_delta_threshold:
            signal_strength += 0.4
            signal_type = 'SELL'
            reason.append(f"Negative delta: {microstructure['cumulative_delta']:.0f}")
        
        # Momentum signal
        if microstructure['price_velocity'] > min_velocity_threshold:
            signal_strength += 0.3
            if signal_type is None:
                signal_type = 'BUY'
            elif signal_type == 'BUY':
                signal_strength += 0.1
            reason.append(f"Positive momentum: {microstructure['price_velocity']:.6f}")
        elif microstructure['price_velocity'] < -min_velocity_threshold:
            signal_strength += 0.3
            if signal_type is None:
                signal_type = 'SELL'
            elif signal_type == 'SELL':
                signal_strength += 0.1
            reason.append(f"Negative momentum: {microstructure['price_velocity']:.6f}")
        
        # Volatility check
        if microstructure['volatility'] > self.config.get('max_volatility', 0.001):
            signal_strength *= 0.5
            reason.append("High volatility - reduced confidence")
        
        # Check if we should close position
        if self.position_type is not None:
            if self.position_type == 'BUY' and signal_type == 'SELL' and signal_strength > 0.6:
                return Signal(
                    timestamp=time.time(),
                    signal_type='CLOSE',
                    strength=signal_strength,
                    price=current_tick.bid,
                    stop_loss=0,
                    take_profit=0,
                    volume=self.position_volume,
                    reason="Reversal signal detected"
                )
            elif self.position_type == 'SELL' and signal_type == 'BUY' and signal_strength > 0.6:
                return Signal(
                    timestamp=time.time(),
                    signal_type='CLOSE',
                    strength=signal_strength,
                    price=current_tick.ask,
                    stop_loss=0,
                    take_profit=0,
                    volume=self.position_volume,
                    reason="Reversal signal detected"
                )
        
        # Generate new signal
        min_strength = self.config.get('min_signal_strength', 0.6)
        
        if signal_type and signal_strength >= min_strength:
            # Calculate SL/TP with minimum distance based on stops level
            atr = microstructure['volatility'] * 10
            
            # Get SL multiplier from config (default 2.0)
            sl_multiplier = self.config.get('sl_multiplier', 2.0)
            
            # Calculate minimum distance based on stops level
            min_distance = self.stops_level * self.symbol_point if self.stops_level > 0 else 0.5
            
            # Ensure SL distance is at least min_distance + some buffer
            sl_distance = max(
                atr * sl_multiplier,  # Use configurable multiplier
                microstructure['avg_spread'] * 5,
                min_distance * 1.2  # 20% buffer above minimum
            )
            
            # Calculate TP based on mode
            tp_mode = self.config.get('tp_mode', 'RiskReward')
            
            if tp_mode == 'FixedDollar':
                # TP based on dollar amount
                tp_dollar = self.config.get('tp_dollar_amount', 0.5)
                volume = self.config.get('default_volume', 0.01)
                
                # Get symbol info for calculation
                symbol_info = mt5.symbol_info(self.symbol)
                if symbol_info and symbol_info.trade_contract_size > 0:
                    # For commodities/metals: Profit = price_diff × volume × contract_size
                    # Therefore: price_diff = profit / (volume × contract_size)
                    contract_size = symbol_info.trade_contract_size
                    tp_distance_raw = tp_dollar / (volume * contract_size)
                    
                    # Ensure TP distance meets minimum required distance
                    if tp_distance_raw < min_distance:
                        tp_distance = min_distance * 1.2  # Use min distance with buffer
                        actual_profit = tp_distance * volume * contract_size
                        logger.warning(f"TP target ${tp_dollar:.2f} too small (needs {tp_distance_raw:.5f}). Using min distance {tp_distance:.5f} (≈${actual_profit:.2f})")
                    else:
                        tp_distance = tp_distance_raw
                        logger.debug(f"TP Mode: FixedDollar (${tp_dollar:.2f}) = {tp_distance:.5f} price distance")
                else:
                    # Fallback to risk:reward if symbol info unavailable
                    tp_distance = sl_distance * self.config.get('risk_reward_ratio', 2.0)
                    logger.warning(f"Failed to get symbol info, using Risk:Reward")
            else:
                # TP based on Risk:Reward ratio (default)
                tp_distance = sl_distance * self.config.get('risk_reward_ratio', 2.0)
            
            if signal_type == 'BUY':
                price = current_tick.ask
                sl = price - sl_distance
                tp = price + tp_distance
            else:
                price = current_tick.bid
                sl = price + sl_distance
                tp = price - tp_distance
            
            logger.debug(f"SL/TP: distance={sl_distance:.5f} (min={min_distance:.5f}), TP distance={tp_distance:.5f}")
            
            return Signal(
                timestamp=time.time(),
                signal_type=signal_type,
                strength=signal_strength,
                price=price,
                stop_loss=sl,
                take_profit=tp,
                volume=self.config.get('default_volume', 0.01),
                reason=" | ".join(reason)
            )
        elif signal_type:
            # Signal exists but not strong enough - log occasionally
            if np.random.random() < 0.1:  # Log 10% of weak signals
                logger.warning(f"⚠️ WEAK SIGNAL: {signal_type} | Strength: {signal_strength:.2f} < {min_strength:.2f}")
                logger.warning(f"   Thresholds: Delta={min_delta_threshold} | Velocity={min_velocity_threshold:.6f}")
                logger.warning(f"   Actuals: Delta={microstructure['cumulative_delta']:.0f} | Velocity={microstructure['price_velocity']:.6f}")
        else:
            # No signal type at all - log very occasionally to show thresholds
            if np.random.random() < 0.01:  # 1% of the time
                logger.info(f"🔍 No signal criteria met. Need: Delta>{min_delta_threshold} OR Velocity>{min_velocity_threshold:.6f}")
                logger.info(f"   Current: Delta={microstructure['cumulative_delta']:.0f} | Velocity={microstructure['price_velocity']:.6f}")
        
        return None
    
    def verify_position_exists(self) -> bool:
        """Check if position actually exists in MT5"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return False
            
            # Check if any position matches our magic number
            magic = self.config.get('magic_number', 2026001)
            for pos in positions:
                if pos.magic == magic:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking position: {e}")
            return False
    
    def execute_signal(self, signal: Signal) -> bool:
        """Execute trading signal with ultra-low latency"""
        start_time = time.perf_counter()
        
        try:
            # For multi-position support, only verify positions exist (don't block new signals)
            if self.position_type and signal.signal_type != 'CLOSE':
                # Verify positions still exist in MT5
                if not self.verify_position_exists():
                    logger.info(f"🔄 All positions closed (SL/TP hit) - Resetting internal state")
                    self.position_type = None
                    self.position_volume = 0.0
                    self.position_price = 0.0
            
            if signal.signal_type == 'CLOSE':
                result = self.close_position()
            elif signal.signal_type == 'BUY':
                result = self.open_position('BUY', signal)
            elif signal.signal_type == 'SELL':
                result = self.open_position('SELL', signal)
            else:
                return False
            
            # Track execution time
            exec_time = (time.perf_counter() - start_time) * 1000  # milliseconds
            self.execution_times.append(exec_time)
            
            if result:
                logger.info(f"✓ Executed {signal.signal_type} | "
                          f"Price: {signal.price:.5f} | "
                          f"Strength: {signal.strength:.2f} | "
                          f"Time: {exec_time:.2f}ms | "
                          f"Reason: {signal.reason}")
            
            return result
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False
    
    def get_total_floating_loss(self) -> float:
        """Calculate total floating loss from all open positions"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None or len(positions) == 0:
                return 0.0
            
            magic = self.config.get('magic_number', 2026001)
            total_loss = 0.0
            
            for pos in positions:
                if pos.magic == magic and pos.profit < 0:
                    total_loss += abs(pos.profit)
            
            return total_loss
            
        except Exception as e:
            logger.error(f"Error calculating floating loss: {e}")
            return 0.0
    
    def get_total_floating_profit(self) -> float:
        """Calculate total floating profit from all open positions"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None or len(positions) == 0:
                return 0.0
            
            magic = self.config.get('magic_number', 2026001)
            total_profit = 0.0
            
            for pos in positions:
                if pos.magic == magic and pos.profit > 0:
                    total_profit += pos.profit
            
            return total_profit
            
        except Exception as e:
            logger.error(f"Error calculating floating profit: {e}")
            return 0.0
    
    def close_all_positions(self, reason: str = "Target reached") -> int:
        """Close all positions with our magic number"""
        try:
            magic = self.config.get('magic_number', 2026001)
            positions = mt5.positions_get(symbol=self.symbol)
            
            if positions is None or len(positions) == 0:
                return 0
            
            closed_count = 0
            total_profit = 0.0
            
            for position in positions:
                if position.magic != magic:
                    continue
                
                # Prepare close request
                close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(self.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask
                
                # Get filling mode from config
                filling_mode_str = self.config.get('filling_mode', 'FOK')
                filling_mode = self.get_filling_mode(filling_mode_str)
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": position.volume,
                    "type": close_type,
                    "position": position.ticket,
                    "price": price,
                    "deviation": self.config.get('slippage', 20),
                    "magic": magic,
                    "comment": f"CloseAll_{reason}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling_mode,
                }
                
                result = mt5.order_send(request)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    closed_count += 1
                    total_profit += position.profit
                    logger.info(f"✓ Closed position #{position.ticket}: Profit=${position.profit:.2f}")
                else:
                    logger.warning(f"Failed to close position #{position.ticket}: {result.retcode} - {result.comment}")
            
            if closed_count > 0:
                logger.info(f"🎯 CLOSE ALL COMPLETE: {closed_count} positions closed | Total Profit: ${total_profit:.2f} | Reason: {reason}")
                self.position_type = None
                self.position_volume = 0.0
                self.position_price = 0.0
            
            return closed_count
            
        except Exception as e:
            logger.error(f"Close all positions error: {e}")
            return 0
    
    def open_position(self, order_type: str, signal: Signal) -> bool:
        """Open new position"""
        # Check floating loss limit
        max_floating_loss = self.config.get('max_floating_loss', 500)
        current_floating_loss = self.get_total_floating_loss()
        
        if current_floating_loss >= max_floating_loss:
            logger.warning(f"⚠️ Max floating loss reached: ${current_floating_loss:.2f} >= ${max_floating_loss:.2f}")
            logger.warning(f"   Skipping new position to protect capital")
            return False
        
        # Check max positions (only count positions with our magic number)
        max_positions = self.config.get('max_positions', 3)
        magic = self.config.get('magic_number', 2026001)
        positions = mt5.positions_get(symbol=self.symbol)
        
        # Count only positions with our magic number
        our_positions_count = 0
        if positions:
            for pos in positions:
                if pos.magic == magic:
                    our_positions_count += 1
        
        if our_positions_count >= max_positions:
            logger.warning(f"⚠️ Max positions limit reached: {our_positions_count}/{max_positions} (Magic: {magic})")
            return False
        
        logger.info(f"📈 Attempting to open {order_type} position... (Floating Loss: ${current_floating_loss:.2f}/${max_floating_loss:.2f})")
        
        try:
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                return False
            
            # Validate TP distance before sending order
            min_required = self.stops_level * self.symbol_point if self.stops_level > 0 else 0.5
            tp_distance = abs(signal.price - signal.take_profit)
            sl_distance = abs(signal.price - signal.stop_loss)
            
            if tp_distance < min_required:
                logger.error(f"❌ TP distance too small: {tp_distance:.5f} < {min_required:.5f}")
                logger.error(f"   This should not happen! Check generate_signal() TP calculation")
                logger.error(f"   TP Mode: {self.config.get('tp_mode', 'RiskReward')}, Target: ${self.config.get('tp_dollar_amount', 0.5):.2f}")
                return False
            
            if sl_distance < min_required:
                logger.error(f"❌ SL distance too small: {sl_distance:.5f} < {min_required:.5f}")
                return False
            
            # Get filling mode from config
            filling_mode_str = self.config.get('filling_mode', 'FOK')
            filling_mode = self.get_filling_mode(filling_mode_str)
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": signal.volume,
                "type": mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": signal.price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "deviation": self.config.get('slippage', 20),
                "magic": self.get_filling_mode(filling_mode_str),
                "comment": f"AvHFTPro2026_{order_type}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Update position tracking (for multi-position support, we track the latest)
                self.position_type = order_type
                self.position_volume = signal.volume
                self.position_price = result.price
                
                # Log with position count (only our magic number)
                magic = self.config.get('magic_number', 2026001)
                positions = mt5.positions_get(symbol=self.symbol)
                pos_count = 0
                if positions:
                    for pos in positions:
                        if pos.magic == magic:
                            pos_count += 1
                
                logger.info(f"✓ Position #{pos_count} opened: {order_type} {signal.volume} @ {result.price:.5f}")
                logger.info(f"   Bot positions: {pos_count}/{self.config.get('max_positions', 3)} | Magic: {magic}")
                
                return True
            else:
                logger.warning(f"Order failed: {result.retcode} - {result.comment}")
                
                # Detailed error info for debugging
                if result.retcode == 10016:  # Invalid stops
                    logger.error(f"Invalid stops error details:")
                    logger.error(f"  Price: {signal.price:.5f}")
                    logger.error(f"  SL: {signal.stop_loss:.5f} (distance: {abs(signal.price - signal.stop_loss):.5f})")
                    logger.error(f"  TP: {signal.take_profit:.5f} (distance: {abs(signal.price - signal.take_profit):.5f})")
                    logger.error(f"  Min required distance: {self.stops_level * self.symbol_point:.5f} ({self.stops_level} points)")
                
                return False
                
        except Exception as e:
            logger.error(f"Open position error: {e}")
            return False
    
    def close_position(self) -> bool:
        """Close current position (only positions with our magic number)"""
        if self.position_type is None:
            return False
        
        try:
            magic = self.config.get('magic_number', 2026001)
            positions = mt5.positions_get(symbol=self.symbol)
            
            if positions is None or len(positions) == 0:
                self.position_type = None
                return False
            
            # Find first position with our magic number
            position = None
            for pos in positions:
                if pos.magic == magic:
                    position = pos
                    break
            
            if position is None:
                logger.warning(f"No position found with magic number {magic}")
                self.position_type = None
                return False
            
            # Prepare close request
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask
            
            # Get filling mode from config
            filling_mode_str = self.config.get('filling_mode', 'FOK')
            filling_mode = self.get_filling_mode(filling_mode_str)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": position.ticket,
                "price": price,
                "deviation": self.config.get('slippage', 20),
                "magic": self.config.get('magic_number', 2026001),
                "comment": "AvHFTPro2026_CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                profit = position.profit
                logger.info(f"✓ Position closed: Profit={profit:.2f}")
                # Record trade to risk_manager
                if self.risk_manager:
                    from risk_manager import TradeRecord
                    trade_record = TradeRecord(
                        timestamp=datetime.now(),
                        symbol=self.symbol,
                        trade_type='CLOSE',
                        volume=position.volume,
                        open_price=position.price_open,
                        close_price=position.price_current,
                        profit=profit,
                        duration=0,
                        reason='Position closed from engine'
                    )
                    self.risk_manager.record_trade(trade_record)
                self.position_type = None
                self.position_volume = 0.0
                return True
            else:
                logger.warning(f"Close failed: {result.retcode} - {result.comment}")
                return False
                
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return False
    
    def data_collection_loop(self):
        """Ultra-fast data collection thread"""
        logger.info("Data collection thread started")
        
        while self.is_running:
            try:
                # Get tick data
                tick = self.get_tick_ultra_fast()
                if tick:
                    self.tick_buffer.append(tick)
                    
                    # Calculate order flow
                    orderflow = self.calculate_order_flow(tick)
                    if orderflow:
                        self.orderflow_buffer.append(orderflow)
                
                # Sleep for minimal time (adjust based on broker tick frequency)
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                logger.error(f"Data collection error: {e}")
                time.sleep(0.1)
    
    def analysis_loop(self):
        """Market analysis and signal generation thread"""
        logger.info("Analysis thread started")
        analysis_count = 0
        last_position_check = time.time()
        
        while self.is_running:
            try:
                # Periodic position sync check (every 5 seconds)
                current_time = time.time()
                if current_time - last_position_check > 5.0:
                    # Check position status and floating loss (only our magic number)
                    magic = self.config.get('magic_number', 2026001)
                    positions = mt5.positions_get(symbol=self.symbol)
                    
                    # Count only our positions
                    pos_count = 0
                    if positions:
                        for pos in positions:
                            if pos.magic == magic:
                                pos_count += 1
                    
                    floating_loss = self.get_total_floating_loss()
                    max_floating = self.config.get('max_floating_loss', 500)
                    
                    # Check floating profit target
                    floating_profit = self.get_total_floating_profit()
                    max_profit_target = self.config.get('max_floating_profit', 1)
                    
                    # Tambahan: close all jika floating profit >= $1
                    if floating_profit >= 1.0:
                        logger.warning(f"🎯 FLOATING PROFIT $1 REACHED: ${floating_profit:.2f} >= $1.00")
                        logger.warning(f"   Closing all positions to secure profit...")
                        closed = self.close_all_positions(reason=f"FloatingProfit_1USD_{floating_profit:.2f}")
                        if closed > 0:
                            logger.info(f"✓ Successfully closed {closed} positions (Floating Profit >= $1)")
                    
                    if pos_count > 0:
                        logger.info(f"📊 Position Status: {pos_count} open (Magic: {magic}) | "
                                  f"Profit: ${floating_profit:.2f} | Loss: ${floating_loss:.2f}/${max_floating:.2f}")
                        
                        # Check if profit target reached - CLOSE ALL
                        if floating_profit >= max_profit_target:
                            logger.warning(f"🎯 PROFIT TARGET REACHED: ${floating_profit:.2f} >= ${max_profit_target:.2f}")
                            logger.warning(f"   Closing all positions to secure profit...")
                            closed = self.close_all_positions(reason=f"Profit_Target_{floating_profit:.2f}")
                            if closed > 0:
                                logger.info(f"✓ Successfully closed {closed} positions")
                        
                        # Reset state if no positions exist
                        if not self.verify_position_exists():
                            logger.info(f"🔄 All positions closed - Resetting state")
                            self.position_type = None
                            self.position_volume = 0.0
                            self.position_price = 0.0
                    
                    last_position_check = current_time
                
                # Analyze market microstructure
                microstructure = self.analyze_microstructure()
                
                if microstructure:
                    analysis_count += 1
                    
                    # Generate signal
                    signal = self.generate_signal(microstructure)
                    
                    if signal:
                        # Add to signal queue
                        if not self.signal_queue.full():
                            self.signal_queue.put(signal)
                            logger.info(f"📊 SIGNAL GENERATED: {signal.signal_type} | "
                                      f"Strength: {signal.strength:.2f} | "
                                      f"Price: {signal.price:.5f} | "
                                      f"Reason: {signal.reason}")
                        else:
                            logger.warning("Signal queue full, skipping signal")
                    else:
                        # Log every 10 analyses with diagnostics
                        if analysis_count % 10 == 0:
                            logger.info(f"⏳ [{analysis_count}] Spread: {microstructure['avg_spread']:.5f} | "
                                      f"Delta: {microstructure['cumulative_delta']:.0f} | "
                                      f"Velocity: {microstructure['price_velocity']:.6f} | "
                                      f"Volatility: {microstructure['volatility']:.5f}")
                        # Log every 50 for summary
                        if analysis_count % 50 == 0:
                            logger.info(f"⏳ Analyzing market... ({analysis_count} analyses, no strong signal yet)")
                else:
                    logger.debug("Waiting for sufficient tick data...")
                
                # Analysis frequency
                time.sleep(self.config.get('analysis_interval', 0.1))  # 100ms
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                time.sleep(1)
    
    def execution_loop(self):
        """Signal execution thread"""
        logger.info("Execution thread started")
        
        while self.is_running:
            try:
                # Get signal from queue
                if not self.signal_queue.empty():
                    signal = self.signal_queue.get(timeout=1)
                    
                    # Execute signal
                    self.execute_signal(signal)
                else:
                    time.sleep(0.01)  # 10ms
                    
            except Exception as e:
                logger.error(f"Execution loop error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start HFT engine"""
        logger.info("=" * 60)
        logger.info("Starting Aventa HFT Pro 2026 Engine")
        logger.info("=" * 60)
        
        if not self.initialize():
            logger.error("Failed to initialize")
            return False
        
        self.is_running = True
        
        # Start threads
        self.data_thread = threading.Thread(target=self.data_collection_loop, daemon=True)
        self.analysis_thread = threading.Thread(target=self.analysis_loop, daemon=True)
        self.execution_thread = threading.Thread(target=self.execution_loop, daemon=True)
        
        self.data_thread.start()
        self.analysis_thread.start()
        self.execution_thread.start()
        
        logger.info("✓ All threads started successfully")
        logger.info(f"  Symbol: {self.symbol}")
        logger.info(f"  Analysis interval: {self.config.get('analysis_interval', 0.1)}s")
        logger.info(f"  Default volume: {self.config.get('default_volume', 0.01)}")
        logger.info(f"  Min signal strength: {self.config.get('min_signal_strength', 0.6)}")
        logger.info(f"  Risk/Reward ratio: {self.config.get('risk_reward_ratio', 2.0)}")
        logger.info(f"  Max Floating Loss: ${self.config.get('max_floating_loss', 500):.2f}")
        logger.info(f"  Take Profit Target: ${self.config.get('max_floating_profit', 1000):.2f} (Close All)")
        logger.info("")
        logger.info("🔍 Waiting for trading signals...")
        logger.info("   Bot will trade when:")
        logger.info("   • Order flow delta exceeds threshold")
        logger.info("   • Price momentum is strong enough")
        logger.info("   • Spread is within limits")
        logger.info("   • Signal strength >= threshold")
        logger.info("")
        
        return True
    
    def stop(self):
        """Stop HFT engine"""
        logger.info("Stopping HFT engine...")
        self.is_running = False
        
        # Wait for threads
        if self.data_thread:
            self.data_thread.join(timeout=5)
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        if self.execution_thread:
            self.execution_thread.join(timeout=5)
        
        # Close any open positions
        if self.position_type:
            self.close_position()
        
        mt5.shutdown()
        logger.info("✓ HFT engine stopped")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if len(self.latency_samples) > 0:
            avg_latency = np.mean(list(self.latency_samples))
            max_latency = np.max(list(self.latency_samples))
            min_latency = np.min(list(self.latency_samples))
        else:
            avg_latency = max_latency = min_latency = 0
        
        if len(self.execution_times) > 0:
            avg_exec_time = np.mean(list(self.execution_times))
            max_exec_time = np.max(list(self.execution_times))
        else:
            avg_exec_time = max_exec_time = 0
        
        return {
            'tick_latency_avg_us': avg_latency,
            'tick_latency_max_us': max_latency,
            'tick_latency_min_us': min_latency,
            'execution_time_avg_ms': avg_exec_time,
            'execution_time_max_ms': max_exec_time,
            'ticks_processed': len(self.tick_buffer),
            'orderflow_samples': len(self.orderflow_buffer),
            'current_position': self.position_type,
            'position_volume': self.position_volume
        }


if __name__ == "__main__":
    # Example configuration
    config = {
        'magic_number': 2026001,
        'default_volume': 0.01,
        'min_delta_threshold': 50,
        'min_velocity_threshold': 0.00001,
        'max_spread': 0.0002,
        'max_volatility': 0.001,
        'min_signal_strength': 0.6,
        'risk_reward_ratio': 2.0,
        'analysis_interval': 0.1,
        'slippage': 20
    }
    
    # Create engine
    engine = UltraLowLatencyEngine(symbol="EURUSD", config=config)
    
    try:
        # Start engine
        if engine.start():
            # Run for demo
            logger.info("Engine running... Press Ctrl+C to stop")
            
            while True:
                time.sleep(10)
                
                # Print performance stats
                stats = engine.get_performance_stats()
                logger.info(f"Performance: "
                          f"Latency={stats['tick_latency_avg_us']:.1f}μs | "
                          f"ExecTime={stats['execution_time_avg_ms']:.2f}ms | "
                          f"Ticks={stats['ticks_processed']}")
                
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
    finally:
        engine.stop()
