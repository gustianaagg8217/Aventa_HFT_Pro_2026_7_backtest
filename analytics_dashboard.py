"""
Aventa HFT Pro 2026 - Real-time Analytics Dashboard
Generate real-time performance charts, equity curves, and trading statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EquityCurveAnalyzer:
    """Analyze and plot equity curve metrics"""
    
    def __init__(self, initial_balance: float = 10000):
        """
        Initialize equity curve analyzer
        
        Args:
            initial_balance: Starting account balance
        """
        self.initial_balance = initial_balance
        self.equity_curve = deque(maxlen=10000)
        self.timestamps = deque(maxlen=10000)
        self.trades = []  # List of (time, pnl, open_price, close_price)
        self.current_balance = initial_balance
    
    def add_balance_update(self, timestamp: datetime, balance: float):
        """Add balance update to equity curve"""
        self.timestamps.append(timestamp)
        self.equity_curve.append(balance)
        self.current_balance = balance
    
    def add_trade(self, timestamp: datetime, pnl: float, entry_price: float, exit_price: float):
        """Record a trade"""
        self.trades.append({
            'time': timestamp,
            'pnl': pnl,
            'entry': entry_price,
            'exit': exit_price,
            'win': pnl > 0
        })
    
    def calculate_drawdown_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate drawdown from peak
        
        Returns:
            (times, drawdown_percentages)
        """
        if len(self.equity_curve) < 2:
            return np.array([]), np.array([])
        
        equity = np.array(list(self.equity_curve))
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        
        return np.array(list(self.timestamps)), drawdown
    
    def calculate_underwater_plot(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate underwater plot (equity below peak)
        
        Returns:
            (times, underwater_values)
        """
        return self.calculate_drawdown_curve()
    
    def get_peak_drawdown(self) -> float:
        """Get maximum drawdown percentage"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        times, drawdowns = self.calculate_drawdown_curve()
        if len(drawdowns) == 0:
            return 0.0
        
        return abs(min(drawdowns))
    
    def get_recovery_time(self) -> Optional[int]:
        """Get bars to recover from max drawdown"""
        if len(self.equity_curve) < 2:
            return None
        
        times, drawdowns = self.calculate_drawdown_curve()
        
        # Find max drawdown location
        max_dd_idx = np.argmin(drawdowns)
        
        # Find when drawdown recovers (crosses below 0 or reaches peak again)
        for i in range(max_dd_idx + 1, len(drawdowns)):
            if drawdowns[i] >= -1:  # Recovery point (within 1%)
                return i - max_dd_idx
        
        return None


class PerformanceAnalyzer:
    """Calculate and track performance metrics"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Array of periodic returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio (higher is better)
        """
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (only considers downside volatility)
        
        Args:
            returns: Array of periodic returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = np.minimum(excess_returns, 0)
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, initial_balance: float) -> float:
        """
        Calculate Calmar ratio (return / max drawdown)
        
        Args:
            returns: Array of period returns
            initial_balance: Starting balance
            
        Returns:
            Calmar ratio
        """
        cumulative_returns = (1 + returns).cumprod()
        max_dd = (cumulative_returns.min() - 1) / 1
        
        if max_dd >= 0:
            return 0.0
        
        total_return = cumulative_returns[-1] - 1
        return total_return / abs(max_dd)
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict]) -> float:
        """Calculate win rate percentage"""
        if not trades:
            return 0.0
        
        wins = sum(1 for t in trades if t['win'])
        return wins / len(trades) * 100
    
    @staticmethod
    def calculate_profit_factor(trades: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not trades:
            return 0.0
        
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_expectancy(trades: List[Dict]) -> float:
        """
        Calculate expectancy (expected profit per trade)
        
        Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
        """
        if not trades:
            return 0.0
        
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]
        
        win_rate = len(wins) / len(trades)
        loss_rate = len(losses) / len(trades)
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        return win_rate * avg_win - loss_rate * avg_loss


class RealTimeCharts:
    """Generate real-time trading charts"""
    
    @staticmethod
    def create_equity_curve_chart(analyzer: EquityCurveAnalyzer, parent_frame) -> FigureCanvasTkAgg:
        """
        Create equity curve chart
        
        Args:
            analyzer: EquityCurveAnalyzer instance
            parent_frame: Tkinter frame to embed chart
            
        Returns:
            FigureCanvasTkAgg widget
        """
        fig = Figure(figsize=(12, 4), dpi=100, facecolor='#1e1e1e')
        ax = fig.add_subplot(111)
        
        if len(analyzer.equity_curve) > 1:
            times = list(analyzer.timestamps)
            balances = list(analyzer.equity_curve)
            
            ax.plot(times, balances, linewidth=2, color='#00ff00', label='Equity Curve')
            ax.fill_between(times, analyzer.initial_balance, balances, 
                           alpha=0.3, color='#00ff00')
            ax.axhline(y=analyzer.initial_balance, color='#ffff00', linestyle='--', 
                       label='Initial Balance', linewidth=1)
            
            # Final balance text
            final_balance = balances[-1]
            pnl = final_balance - analyzer.initial_balance
            pnl_pct = pnl / analyzer.initial_balance * 100
            
            ax.text(0.02, 0.98, f'Final: ${final_balance:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8),
                   color='#00ff00' if pnl > 0 else '#ff0000')
        
        ax.set_facecolor('#0a0a0a')
        ax.grid(True, alpha=0.2, color='#666666')
        ax.set_xlabel('Time')
        ax.set_ylabel('Balance ($)')
        ax.set_title('Equity Curve')
        ax.legend(loc='upper left')
        
        # Rotate x-axis labels
        fig.autofmt_xdate()
        
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        
        return canvas
    
    @staticmethod
    def create_drawdown_chart(analyzer: EquityCurveAnalyzer, parent_frame) -> FigureCanvasTkAgg:
        """
        Create underwater drawdown chart
        
        Args:
            analyzer: EquityCurveAnalyzer instance
            parent_frame: Tkinter frame to embed chart
            
        Returns:
            FigureCanvasTkAgg widget
        """
        fig = Figure(figsize=(12, 3), dpi=100, facecolor='#1e1e1e')
        ax = fig.add_subplot(111)
        
        times, drawdowns = analyzer.calculate_underwater_plot()
        
        if len(drawdowns) > 1:
            ax.fill_between(times, 0, drawdowns, alpha=0.5, color='#ff3333')
            ax.plot(times, drawdowns, linewidth=1, color='#ff0000')
            
            # Max drawdown text
            max_dd = analyzer.get_peak_drawdown()
            ax.text(0.02, 0.98, f'Max Drawdown: {-max_dd:.2f}%',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8),
                   color='#ffff00')
        
        ax.set_facecolor('#0a0a0a')
        ax.grid(True, alpha=0.2, color='#666666')
        ax.set_xlabel('Time')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Underwater Plot (Drawdown from Peak)')
        
        fig.autofmt_xdate()
        
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        
        return canvas
    
    @staticmethod
    def create_trades_analysis_chart(trades: List[Dict], parent_frame) -> FigureCanvasTkAgg:
        """
        Create bar chart of trades (wins vs losses)
        
        Args:
            trades: List of trade dictionaries
            parent_frame: Tkinter frame
            
        Returns:
            FigureCanvasTkAgg widget
        """
        fig = Figure(figsize=(12, 3), dpi=100, facecolor='#1e1e1e')
        ax = fig.add_subplot(111)
        
        if not trades:
            ax.text(0.5, 0.5, 'No trades yet', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='#ffffff')
        else:
            # Separate wins and losses
            indices = np.arange(len(trades))
            pnls = [t['pnl'] for t in trades]
            colors = ['#00ff00' if pnl > 0 else '#ff0000' for pnl in pnls]
            
            ax.bar(indices, pnls, color=colors, alpha=0.7, width=0.8)
            ax.axhline(y=0, color='#ffffff', linestyle='-', linewidth=0.5)
            
            # Statistics
            total_pnl = sum(pnls)
            avg_pnl = np.mean(pnls)
            win_rate = PerformanceAnalyzer.calculate_win_rate(trades)
            
            stats_text = f'Total: ${total_pnl:+.2f} | Avg: ${avg_pnl:+.2f} | Win Rate: {win_rate:.1f}%'
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8),
                   color='#ffffff')
        
        ax.set_facecolor('#0a0a0a')
        ax.grid(True, alpha=0.2, axis='y', color='#666666')
        ax.set_xlabel('Trade #')
        ax.set_ylabel('P&L ($)')
        ax.set_title('Trade by Trade Analysis')
        
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        
        return canvas
    
    @staticmethod
    def create_performance_heatmap(daily_returns: Dict[str, float], parent_frame) -> FigureCanvasTkAgg:
        """
        Create performance heatmap by month/day
        
        Args:
            daily_returns: Dictionary of {date: return_percentage}
            parent_frame: Tkinter frame
            
        Returns:
            FigureCanvasTkAgg widget
        """
        fig = Figure(figsize=(12, 3), dpi=100, facecolor='#1e1e1e')
        ax = fig.add_subplot(111)
        
        if not daily_returns:
            ax.text(0.5, 0.5, 'No performance data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='#ffffff')
        else:
            dates = list(daily_returns.keys())
            returns = list(daily_returns.values())
            colors = ['#00ff00' if r > 0 else '#ff0000' for r in returns]
            
            indices = np.arange(len(returns))
            ax.bar(indices, returns, color=colors, alpha=0.7, width=0.8)
            ax.axhline(y=0, color='#ffffff', linestyle='-', linewidth=0.5)
            
            # Statistics
            avg_return = np.mean(returns)
            best_day = max(returns)
            worst_day = min(returns)
            
            stats_text = f'Avg: {avg_return:+.2f}% | Best: {best_day:+.2f}% | Worst: {worst_day:+.2f}%'
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8),
                   color='#ffffff')
        
        ax.set_facecolor('#0a0a0a')
        ax.grid(True, alpha=0.2, axis='y', color='#666666')
        ax.set_xlabel('Day')
        ax.set_ylabel('Daily Return (%)')
        ax.set_title('Daily Performance')
        
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        
        return canvas


class DashboardWidget(ttk.Frame):
    """Embeddable dashboard widget for real-time metrics display"""
    
    def __init__(self, parent, analyzer: EquityCurveAnalyzer = None):
        """
        Initialize dashboard widget
        
        Args:
            parent: Parent Tkinter widget
            analyzer: EquityCurveAnalyzer instance
        """
        super().__init__(parent)
        
        self.analyzer = analyzer
        
        # Create metric labels
        self.metric_frame = ttk.LabelFrame(self, text="Live Metrics")
        self.metric_frame.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Metrics dictionary
        self.metrics = {}
        self._create_metric_labels()
    
    def _create_metric_labels(self):
        """Create metric display labels"""
        metric_names = [
            'Balance:', 'P&L:', 'P&L %:', 'Max DD:', 'Sharpe:', 
            'Trades:', 'Win Rate:', 'Profit Factor:'
        ]
        
        for i, name in enumerate(metric_names):
            row = i // 4
            col = i % 4
            
            label = ttk.Label(self.metric_frame, text=name)
            label.grid(row=row, column=col*2, sticky='w', padx=5, pady=2)
            
            value_label = ttk.Label(self.metric_frame, text="--", foreground='#00ff00')
            value_label.grid(row=row, column=col*2+1, sticky='e', padx=5, pady=2)
            
            self.metrics[name] = value_label
    
    def update_metrics(self, analyzer: EquityCurveAnalyzer, trades: List[Dict]):
        """Update all metric displays"""
        if not analyzer or len(analyzer.equity_curve) == 0:
            return
        
        current_balance = list(analyzer.equity_curve)[-1]
        pnl = current_balance - analyzer.initial_balance
        pnl_pct = pnl / analyzer.initial_balance * 100
        max_dd = analyzer.get_peak_drawdown()
        
        sharpe = 0  # Calculate from returns if available
        win_rate = PerformanceAnalyzer.calculate_win_rate(trades)
        profit_factor = PerformanceAnalyzer.calculate_profit_factor(trades)
        
        self.metrics['Balance:'].config(text=f'${current_balance:.2f}')
        self.metrics['P&L:'].config(text=f'${pnl:+.2f}')
        self.metrics['P&L %:'].config(text=f'{pnl_pct:+.1f}%')
        self.metrics['Max DD:'].config(text=f'{-max_dd:+.1f}%')
        self.metrics['Sharpe:'].config(text=f'{sharpe:.2f}')
        self.metrics['Trades:'].config(text=f'{len(trades)}')
        self.metrics['Win Rate:'].config(text=f'{win_rate:.1f}%')
        self.metrics['Profit Factor:'].config(text=f'{profit_factor:.2f}')
        
        # Update colors based on P&L
        pnl_color = '#00ff00' if pnl > 0 else '#ff0000'
        self.metrics['P&L:'].config(foreground=pnl_color)
        self.metrics['P&L %:'].config(foreground=pnl_color)
