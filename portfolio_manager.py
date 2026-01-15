"""
Aventa HFT Pro 2026 - Multi-Symbol Portfolio Management
Manage multiple symbols simultaneously with correlated risk allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class SymbolPosition:
    """Position data for a single symbol"""
    symbol: str
    quantity: float = 0
    entry_price: float = 0
    current_price: float = 0
    entry_time: datetime = None
    
    # Risk management
    stop_loss: float = 0
    take_profit: float = 0
    max_loss: float = 0
    trailing_stop: float = 0
    
    # P&L tracking
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    
    def update_price(self, price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = price
        if self.quantity != 0 and self.entry_price != 0:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
    
    @property
    def is_open(self) -> bool:
        """Check if position is open"""
        return self.quantity != 0
    
    @property
    def pnl_percent(self) -> float:
        """Get P&L percentage"""
        if self.entry_price == 0:
            return 0
        return (self.current_price - self.entry_price) / self.entry_price * 100
    
    def close(self, price: float, fee: float = 0):
        """Close position and record realized P&L"""
        pnl = (price - self.entry_price) * self.quantity - fee
        self.realized_pnl = pnl
        self.unrealized_pnl = 0
        self.quantity = 0
        self.entry_price = 0


@dataclass
class PortfolioMetrics:
    """Portfolio-level performance metrics"""
    total_value: float = 0
    total_cash: float = 0
    total_exposure: float = 0
    net_pnl: float = 0
    portfolio_return_pct: float = 0
    
    # Risk metrics
    portfolio_volatility: float = 0
    max_drawdown: float = 0
    correlation_average: float = 0
    diversification_ratio: float = 0
    
    # Position metrics
    num_open_positions: int = 0
    concentration_ratio: float = 0
    
    # Performance metrics
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    profit_factor: float = 0
    win_rate: float = 0


class SymbolCorrelationAnalyzer:
    """Analyze symbol correlations for diversification"""
    
    def __init__(self, lookback_periods: int = 100):
        """
        Initialize correlation analyzer
        
        Args:
            lookback_periods: Number of periods for correlation calculation
        """
        self.lookback_periods = lookback_periods
        self.returns_history = {}  # {symbol: [returns]}
        self.correlation_matrix = None
    
    def add_returns(self, symbol: str, returns: List[float]):
        """
        Add returns for a symbol
        
        Args:
            symbol: Trading symbol
            returns: List of returns
        """
        if symbol not in self.returns_history:
            self.returns_history[symbol] = []
        
        self.returns_history[symbol].extend(returns)
        self.returns_history[symbol] = self.returns_history[symbol][-self.lookback_periods:]
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between all symbols
        
        Returns:
            Correlation matrix DataFrame
        """
        if not self.returns_history:
            return None
        
        # Align returns to same length
        min_length = min(len(r) for r in self.returns_history.values())
        aligned_returns = {
            symbol: np.array(returns[-min_length:])
            for symbol, returns in self.returns_history.items()
        }
        
        df_returns = pd.DataFrame(aligned_returns)
        self.correlation_matrix = df_returns.corr()
        
        return self.correlation_matrix
    
    def get_average_correlation(self, symbol: str) -> float:
        """Get average correlation of symbol with others"""
        if self.correlation_matrix is None:
            return 0
        
        if symbol not in self.correlation_matrix.columns:
            return 0
        
        correlations = self.correlation_matrix[symbol]
        other_symbols = correlations.drop(symbol)
        
        return other_symbols.mean()
    
    def find_uncorrelated_symbols(self, threshold: float = 0.3) -> List[str]:
        """
        Find symbols with low correlation (good for diversification)
        
        Args:
            threshold: Maximum average correlation
            
        Returns:
            List of uncorrelated symbols
        """
        if self.correlation_matrix is None:
            return []
        
        uncorrelated = []
        for symbol in self.correlation_matrix.columns:
            avg_corr = self.get_average_correlation(symbol)
            if abs(avg_corr) < threshold:
                uncorrelated.append(symbol)
        
        return uncorrelated


class PortfolioRiskManager:
    """Manage portfolio-level risk"""
    
    def __init__(self, initial_balance: float, config: Optional[Dict] = None):
        """
        Initialize portfolio risk manager
        
        Args:
            initial_balance: Starting account balance
            config: Configuration dictionary
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.config = config or {}
        
        # Risk limits
        self.max_total_exposure = self.config.get('max_total_exposure', 5.0)  # 5x leverage
        self.max_single_symbol = self.config.get('max_single_symbol', 0.20)  # 20% per symbol
        self.max_sector_exposure = self.config.get('max_sector_exposure', 0.30)  # 30% per sector
        self.max_daily_loss = self.config.get('max_daily_loss', self.initial_balance * 0.05)  # 5%
        self.max_portfolio_dd = self.config.get('max_portfolio_dd', 0.15)  # 15% max drawdown
        
        # Tracking
        self.daily_pnl = 0
        self.peak_balance = initial_balance
        self.positions = {}  # {symbol: SymbolPosition}
        self.correlation_analyzer = SymbolCorrelationAnalyzer()
    
    def can_open_position(self, symbol: str, position_size: float, 
                         current_price: float) -> Tuple[bool, str]:
        """
        Check if new position can be opened
        
        Args:
            symbol: Trading symbol
            position_size: Position size (quantity)
            current_price: Current symbol price
            
        Returns:
            (can_open, reason)
        """
        # Calculate position value
        position_value = abs(position_size * current_price)
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            return False, "Daily loss limit exceeded"
        
        # Check portfolio drawdown
        current_dd = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_dd > self.max_portfolio_dd:
            return False, f"Portfolio drawdown limit exceeded ({current_dd:.1%})"
        
        # Calculate current exposure
        current_exposure = sum(
            abs(pos.quantity * pos.current_price) for pos in self.positions.values()
        ) / self.current_balance
        
        # Check total exposure limit
        new_exposure = current_exposure + position_value / self.current_balance
        if new_exposure > self.max_total_exposure:
            return False, f"Total exposure limit exceeded ({new_exposure:.2f}x)"
        
        # Check single symbol limit
        symbol_exposure = position_value / self.current_balance
        if symbol_exposure > self.max_single_symbol:
            return False, f"Single symbol exposure limit exceeded ({symbol_exposure:.1%})"
        
        # Check if symbol already has position
        if symbol in self.positions and self.positions[symbol].is_open:
            return False, f"Position already open for {symbol}"
        
        return True, "OK"
    
    def allocate_risk(self, symbols: List[str], risk_pct_per_trade: float = 1.0) -> Dict[str, float]:
        """
        Allocate risk across multiple symbols (Kelly Criterion variant)
        
        Args:
            symbols: List of symbols to trade
            risk_pct_per_trade: Risk percentage per trade
            
        Returns:
            Dictionary of {symbol: allocation_ratio}
        """
        if not symbols:
            return {}
        
        # Equal allocation (can be enhanced with machine learning)
        base_allocation = 1.0 / len(symbols)
        allocation = {symbol: base_allocation for symbol in symbols}
        
        # Adjust for correlation (reduce allocation for correlated symbols)
        for symbol in symbols:
            avg_corr = self.correlation_analyzer.get_average_correlation(symbol)
            # Reduce allocation if highly correlated
            correlation_factor = 1.0 - (max(0, avg_corr) * 0.3)  # Max 30% reduction
            allocation[symbol] *= correlation_factor
        
        # Normalize to sum to 1
        total = sum(allocation.values())
        allocation = {k: v/total for k, v in allocation.items()}
        
        return allocation
    
    def update_metrics(self) -> PortfolioMetrics:
        """Calculate current portfolio metrics"""
        metrics = PortfolioMetrics()
        
        # Position values
        total_position_value = 0
        total_position_pnl = 0
        
        for symbol, position in self.positions.items():
            if position.is_open:
                position_value = position.quantity * position.current_price
                total_position_value += position_value
                total_position_pnl += position.unrealized_pnl
                metrics.num_open_positions += 1
        
        metrics.total_exposure = total_position_value / self.current_balance
        metrics.total_value = self.current_balance + total_position_pnl
        metrics.total_cash = self.current_balance - total_position_value
        metrics.net_pnl = total_position_pnl
        metrics.portfolio_return_pct = (metrics.total_value - self.initial_balance) / self.initial_balance * 100
        
        # Concentration ratio (Herfindahl index)
        if total_position_value > 0:
            weights = [position.quantity * position.current_price / total_position_value 
                      for position in self.positions.values() if position.is_open]
            metrics.concentration_ratio = sum(w**2 for w in weights)
        
        # Correlation metrics
        self.correlation_analyzer.calculate_correlation_matrix()
        correlations = []
        for symbol in self.positions.keys():
            avg_corr = self.correlation_analyzer.get_average_correlation(symbol)
            if not np.isnan(avg_corr):
                correlations.append(avg_corr)
        
        if correlations:
            metrics.correlation_average = np.mean(correlations)
            metrics.diversification_ratio = 1 - metrics.concentration_ratio
        
        # Drawdown
        metrics.max_drawdown = max(0, (self.peak_balance - self.current_balance) / self.peak_balance)
        
        return metrics
    
    def close_position(self, symbol: str, price: float, fee: float = 0):
        """Close position and update balance"""
        if symbol in self.positions and self.positions[symbol].is_open:
            pos = self.positions[symbol]
            pnl = (price - pos.entry_price) * pos.quantity - fee
            
            self.current_balance += pnl
            self.daily_pnl += pnl
            
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
            
            pos.close(price, fee)
            logger.info(f"Closed {symbol} position: P&L = ${pnl:+.2f}")


class MultiSymbolPortfolio:
    """Manage trading across multiple symbols"""
    
    def __init__(self, initial_balance: float = 10000, config: Optional[Dict] = None):
        """
        Initialize multi-symbol portfolio
        
        Args:
            initial_balance: Starting balance
            config: Configuration
        """
        self.initial_balance = initial_balance
        self.config = config or {}
        
        self.risk_manager = PortfolioRiskManager(initial_balance, config)
        self.positions = self.risk_manager.positions
        self.metrics = None
    
    def open_position(self, symbol: str, quantity: float, entry_price: float,
                     stop_loss: float, take_profit: float) -> bool:
        """
        Open new position
        
        Args:
            symbol: Trading symbol
            quantity: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if position opened successfully
        """
        # Check risk limits
        can_open, reason = self.risk_manager.can_open_position(symbol, quantity, entry_price)
        if not can_open:
            logger.warning(f"Cannot open {symbol}: {reason}")
            return False
        
        # Create position
        position = SymbolPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now()
        )
        
        self.positions[symbol] = position
        logger.info(f"Opened {symbol} position: Qty={quantity}, Entry=${entry_price:.5f}")
        
        return True
    
    def update_prices(self, price_data: Dict[str, float]):
        """
        Update current prices and check for exits
        
        Args:
            price_data: Dictionary of {symbol: current_price}
        """
        for symbol, price in price_data.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                position.update_price(price)
                
                # Check stop loss
                if position.quantity > 0 and price <= position.stop_loss:
                    self.risk_manager.close_position(symbol, position.stop_loss)
                
                # Check take profit
                if position.quantity > 0 and price >= position.take_profit:
                    self.risk_manager.close_position(symbol, position.take_profit)
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        self.metrics = self.risk_manager.update_metrics()
        
        summary = {
            'total_value': self.metrics.total_value,
            'total_pnl': self.metrics.net_pnl,
            'pnl_percent': self.metrics.portfolio_return_pct,
            'num_positions': self.metrics.num_open_positions,
            'total_exposure': f"{self.metrics.total_exposure:.2f}x",
            'max_drawdown': f"{self.metrics.max_drawdown:.2f}%",
            'concentration': f"{self.metrics.concentration_ratio:.2f}",
            'avg_correlation': f"{self.metrics.correlation_average:.2f}",
            'positions': {}
        }
        
        # Add position details
        for symbol, position in self.positions.items():
            if position.is_open:
                summary['positions'][symbol] = {
                    'quantity': position.quantity,
                    'entry': position.entry_price,
                    'current': position.current_price,
                    'pnl': position.unrealized_pnl,
                    'pnl_pct': position.pnl_percent,
                    'open_time': position.entry_time.isoformat()
                }
        
        return summary
    
    def rebalance_portfolio(self, target_symbols: List[str]):
        """
        Rebalance portfolio to target symbols with risk allocation
        
        Args:
            target_symbols: List of symbols to maintain exposure to
        """
        logger.info(f"Rebalancing portfolio to {target_symbols}")
        
        # Close positions not in target list
        symbols_to_close = [s for s in self.positions.keys() if s not in target_symbols]
        for symbol in symbols_to_close:
            if self.positions[symbol].is_open:
                self.risk_manager.close_position(symbol, self.positions[symbol].current_price)
        
        # Allocate risk across targets
        allocation = self.risk_manager.allocate_risk(target_symbols)
        logger.info(f"Risk allocation: {allocation}")
    
    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get current correlation matrix"""
        return self.risk_manager.correlation_analyzer.correlation_matrix
