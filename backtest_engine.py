"""
Aventa HFT Pro 2026 - Complete Backtesting Engine
Comprehensive backtesting with proper OHLCV simulation, slippage, commissions
"""

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a single trade during backtest"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    symbol: str = "EURUSD"
    trade_type: str = "BUY"  # BUY or SELL
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    volume: float = 0.01
    stop_loss: float = 0.0
    take_profit: float = 0.0
    gross_pnl: float = 0.0
    pnl_after_commission: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    bars_held: int = 0
    win: bool = False
    reason: str = ""


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    best_trade: float = 0.0
    worst_trade: float = 0.0
    
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    avg_trade_duration: float = 0.0
    
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)


class BacktestEngine:
    """Professional backtesting engine for HFT strategies"""
    
    def __init__(self, symbol: str = "EURUSD", initial_balance: float = 10000.0):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        self.trades: List[BacktestTrade] = []
        self.equity_curve: deque = deque(maxlen=100000)
        self.drawdown_curve: deque = deque(maxlen=100000)
        
        self.open_positions: Dict[int, BacktestTrade] = {}  # position_id -> trade
        self.position_counter = 0
        
    def load_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                  timeframe: int = mt5.TIMEFRAME_M1) -> Optional[pd.DataFrame]:
        """Load historical OHLCV data from MT5"""
        try:
            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return None
            
            # Calculate bars needed
            bars_needed = (end_date - start_date).days * 24 * 60  # For M1
            
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                logger.error(f"No data retrieved for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            logger.info(f"✓ Loaded {len(df)} bars for {symbol}")
            mt5.shutdown()
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def run_backtest(self, df: pd.DataFrame, strategy_func, 
                     commission: float = 0.0002, slippage: float = 0.00001) -> BacktestMetrics:
        """
        Run backtest with given strategy function
        
        Args:
            df: OHLCV dataframe with columns [open, high, low, close, volume]
            strategy_func: Function that returns signal (1=BUY, -1=SELL, 0=HOLD)
            commission: Commission per trade (0.02% = 0.0002)
            slippage: Slippage per trade (0.1 pips for EURUSD)
        
        Returns:
            BacktestMetrics object with performance statistics
        """
        
        logger.info(f"Starting backtest from {df.index[0]} to {df.index[-1]}")
        logger.info(f"Initial balance: ${self.initial_balance:.2f}")
        
        self.trades = []
        self.equity_curve.clear()
        self.drawdown_curve.clear()
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.open_positions = {}
        self.position_counter = 0
        
        current_position = None
        current_position_id = None
        
        # Iterate through bars
        for i in range(1, len(df)):
            current_bar = df.iloc[i]
            previous_bar = df.iloc[i - 1]
            timestamp = df.index[i]
            
            # Get signal from strategy
            signal = strategy_func(df.iloc[:i])  # Pass all previous data
            
            # Close existing position on opposite signal
            if current_position is not None:
                if (current_position.trade_type == "BUY" and signal == -1) or \
                   (current_position.trade_type == "SELL" and signal == 1):
                    self._close_position(current_position_id, current_bar, timestamp, "Signal reversal")
                    current_position = None
                    current_position_id = None
            
            # Open new position on signal
            if current_position is None and signal != 0:
                trade_type = "BUY" if signal == 1 else "SELL"
                entry_price = current_bar['close']
                
                # Calculate position size (fixed for now)
                volume = 0.01
                
                # Calculate SL/TP
                atr = self._calculate_atr(df.iloc[:i], period=14)
                stop_loss = entry_price - atr * 2 if signal == 1 else entry_price + atr * 2
                take_profit = entry_price + atr * 4 if signal == 1 else entry_price - atr * 4
                
                self.position_counter += 1
                current_position_id = self.position_counter
                
                current_position = BacktestTrade(
                    entry_time=timestamp,
                    symbol=self.symbol,
                    trade_type=trade_type,
                    entry_price=entry_price,
                    volume=volume,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                self.open_positions[current_position_id] = current_position
            
            # Check for TP/SL hits
            if current_position is not None:
                if current_position.trade_type == "BUY":
                    if current_bar['high'] >= current_position.take_profit:
                        self._close_position(current_position_id, current_bar, timestamp, "TP Hit")
                        current_position = None
                        current_position_id = None
                    elif current_bar['low'] <= current_position.stop_loss:
                        self._close_position(current_position_id, current_bar, timestamp, "SL Hit")
                        current_position = None
                        current_position_id = None
                else:  # SELL
                    if current_bar['low'] <= current_position.take_profit:
                        self._close_position(current_position_id, current_bar, timestamp, "TP Hit")
                        current_position = None
                        current_position_id = None
                    elif current_bar['high'] >= current_position.stop_loss:
                        self._close_position(current_position_id, current_bar, timestamp, "SL Hit")
                        current_position = None
                        current_position_id = None
            
            # Update equity curve
            floating_pnl = self._calculate_floating_pnl(current_bar)
            equity = self.current_balance + floating_pnl
            self.equity_curve.append(equity)
            self.drawdown_curve.append(self._calculate_drawdown(equity))
        
        # Close any remaining position
        if current_position is not None:
            last_bar = df.iloc[-1]
            last_timestamp = df.index[-1]
            self._close_position(current_position_id, last_bar, last_timestamp, "End of backtest")
        
        # Calculate metrics
        metrics = self._calculate_metrics(df)
        
        logger.info(f"✓ Backtest complete: {metrics.total_trades} trades")
        logger.info(f"  Net Profit: ${metrics.net_profit:.2f} ({metrics.total_return_pct:.2f}%)")
        logger.info(f"  Win Rate: {metrics.win_rate:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        
        return metrics
    
    def _close_position(self, position_id: int, close_bar, timestamp: datetime, reason: str):
        """Close a specific position"""
        if position_id not in self.open_positions:
            return
        
        trade = self.open_positions[position_id]
        trade.exit_time = timestamp
        trade.exit_price = close_bar['close']
        trade.bars_held = len(self.equity_curve) - 1  # Approximate
        trade.reason = reason
        
        # Calculate PnL
        if trade.trade_type == "BUY":
            gross_pnl = (trade.exit_price - trade.entry_price) * trade.volume * 100000  # For 1 lot
        else:
            gross_pnl = (trade.entry_price - trade.exit_price) * trade.volume * 100000
        
        # Apply commission and slippage
        commission = abs(trade.entry_price * trade.volume * 100000 * 0.0002)  # 0.02% entry
        commission += abs(trade.exit_price * trade.volume * 100000 * 0.0002)   # 0.02% exit
        slippage = abs(trade.entry_price - close_bar['close']) * trade.volume * 100000 * 0.5
        
        trade.gross_pnl = gross_pnl
        trade.commission = commission
        trade.slippage = slippage
        trade.pnl_after_commission = gross_pnl - commission - slippage
        trade.win = trade.pnl_after_commission > 0
        
        # Update balance
        self.current_balance += trade.pnl_after_commission
        
        # Update peak for drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        self.trades.append(trade)
        del self.open_positions[position_id]
    
    def _calculate_floating_pnl(self, current_bar) -> float:
        """Calculate total floating PnL from open positions"""
        pnl = 0.0
        for trade in self.open_positions.values():
            if trade.trade_type == "BUY":
                pnl += (current_bar['close'] - trade.entry_price) * trade.volume * 100000
            else:
                pnl += (trade.entry_price - current_bar['close']) * trade.volume * 100000
        return pnl
    
    def _calculate_drawdown(self, current_equity: float) -> float:
        """Calculate current drawdown percentage"""
        if self.peak_balance == 0:
            return 0.0
        return ((self.peak_balance - current_equity) / self.peak_balance) * 100
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(df) < period:
            return 0.01
        
        high_low = df['high'].iloc[-period:] - df['low'].iloc[-period:]
        high_close = abs(df['high'].iloc[-period:] - df['close'].iloc[-period:].shift())
        low_close = abs(df['low'].iloc[-period:] - df['close'].iloc[-period:].shift())
        
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = tr.mean()
        
        return atr if atr > 0 else 0.01
    
    def _calculate_metrics(self, df: pd.DataFrame) -> BacktestMetrics:
        """Calculate comprehensive performance metrics"""
        metrics = BacktestMetrics()
        
        if len(self.trades) == 0:
            metrics.initial_balance = self.initial_balance
            metrics.final_balance = self.current_balance
            return metrics
        
        metrics.total_trades = len(self.trades)
        metrics.winning_trades = sum(1 for t in self.trades if t.win)
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades
        metrics.win_rate = (metrics.winning_trades / metrics.total_trades * 100) if metrics.total_trades > 0 else 0
        
        # Profit metrics
        metrics.gross_profit = sum(t.gross_pnl for t in self.trades if t.gross_pnl > 0)
        metrics.gross_loss = sum(t.gross_pnl for t in self.trades if t.gross_pnl < 0)
        metrics.total_commission = sum(t.commission for t in self.trades)
        metrics.total_slippage = sum(t.slippage for t in self.trades)
        metrics.net_profit = self.current_balance - self.initial_balance
        
        metrics.profit_factor = metrics.gross_profit / abs(metrics.gross_loss) if metrics.gross_loss != 0 else 0
        
        if metrics.winning_trades > 0:
            metrics.avg_win = metrics.gross_profit / metrics.winning_trades
        if metrics.losing_trades > 0:
            metrics.avg_loss = metrics.gross_loss / metrics.losing_trades
        
        # Win/Loss streaks
        max_consecutive = 0
        current_consecutive = 0
        for trade in self.trades:
            if trade.win:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        metrics.max_consecutive_losses = max_consecutive
        
        # Best/Worst trade
        if len(self.trades) > 0:
            metrics.best_trade = max(t.pnl_after_commission for t in self.trades)
            metrics.worst_trade = min(t.pnl_after_commission for t in self.trades)
        
        # Return metrics
        metrics.initial_balance = self.initial_balance
        metrics.final_balance = self.current_balance
        metrics.total_return = metrics.net_profit
        metrics.total_return_pct = (metrics.net_profit / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # Drawdown
        equity_array = np.array(list(self.equity_curve))
        running_max = np.maximum.accumulate(equity_array)
        drawdown = ((equity_array - running_max) / running_max) * 100
        metrics.max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        metrics.max_drawdown_pct = abs(metrics.max_drawdown)
        
        # Sharpe Ratio
        if len(self.trades) > 0:
            returns = np.array([t.pnl_after_commission / self.initial_balance for t in self.trades])
            if np.std(returns) > 0:
                metrics.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            # Sortino Ratio (only downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    metrics.sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252)
        
        # Calmar Ratio
        if metrics.max_drawdown_pct != 0:
            metrics.calmar_ratio = (metrics.total_return_pct / metrics.max_drawdown_pct)
        
        # Average trade duration
        if len(self.trades) > 0:
            durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in self.trades]
            metrics.avg_trade_duration = np.mean(durations)
        
        metrics.equity_curve = list(self.equity_curve)
        metrics.drawdown_curve = list(self.drawdown_curve)
        metrics.timestamps = list(df.index)
        
        return metrics
    
    def export_results(self, filename: str = "backtest_results.csv"):
        """Export trade results to CSV"""
        data = []
        for trade in self.trades:
            data.append({
                'Entry Time': trade.entry_time,
                'Exit Time': trade.exit_time,
                'Type': trade.trade_type,
                'Volume': trade.volume,
                'Entry Price': trade.entry_price,
                'Exit Price': trade.exit_price,
                'SL': trade.stop_loss,
                'TP': trade.take_profit,
                'Gross PnL': trade.gross_pnl,
                'Commission': trade.commission,
                'Slippage': trade.slippage,
                'Net PnL': trade.pnl_after_commission,
                'Win': trade.win,
                'Reason': trade.reason
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"✓ Results exported to {filename}")


# Example usage
if __name__ == "__main__":
    engine = BacktestEngine(symbol="EURUSD", initial_balance=10000)
    
    # Example strategy function
    def simple_strategy(df):
        """Simple moving average crossover"""
        if len(df) < 20:
            return 0
        
        sma_5 = df['close'].iloc[-5:].mean()
        sma_20 = df['close'].iloc[-20:].mean()
        
        if sma_5 > sma_20:
            return 1  # BUY
        elif sma_5 < sma_20:
            return -1  # SELL
        else:
            return 0  # HOLD
    
    # Load data and run backtest
    df = engine.load_data("EURUSD", 
                         datetime(2025, 11, 1), 
                         datetime(2025, 12, 31))
    
    if df is not None:
        metrics = engine.run_backtest(df, simple_strategy)
        engine.export_results("backtest_results.csv")
