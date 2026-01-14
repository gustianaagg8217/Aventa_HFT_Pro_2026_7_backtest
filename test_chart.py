"""
Test script for real-time performance chart
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from collections import deque
from datetime import datetime, timedelta
import random
import threading
import time

class ChartTest:
    def __init__(self, root):
        self.root = root
        self.root.title("Performance Chart Test")
        self.root.geometry("900x700")
        
        # Chart data
        self.chart_data = {
            'timestamps': deque(maxlen=300),
            'pnl': deque(maxlen=300),
            'trades': deque(maxlen=300),
            'win_rate': deque(maxlen=300),
            'latency': deque(maxlen=300)
        }
        
        self.is_running = False
        self.create_gui()
        
    def create_gui(self):
        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Start", command=self.start).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)
        
        # Chart frame
        chart_frame = ttk.LabelFrame(self.root, text="Performance Chart", padding="10")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), facecolor='white')
        self.fig.subplots_adjust(hspace=0.4)
        
        # Create 3 subplots
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        
        # Configure subplots
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.grid(True, alpha=0.3)
        
        # Setup subplot 1 - PnL
        self.ax1.set_title('Daily PnL ($)', fontsize=10)
        self.ax1.set_ylabel('PnL', fontsize=8)
        self.line_pnl, = self.ax1.plot([], [], color='green', linewidth=2, label='PnL')
        self.ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        self.ax1.legend(loc='upper left', fontsize=8)
        
        # Setup subplot 2 - Win Rate & Trades
        self.ax2.set_title('Win Rate & Trade Count', fontsize=10)
        self.ax2.set_ylabel('Win Rate (%)', color='green', fontsize=8)
        self.line_winrate, = self.ax2.plot([], [], color='green', linewidth=2, label='Win Rate')
        self.ax2_twin = self.ax2.twinx()
        self.ax2_twin.set_ylabel('Trades', color='blue', fontsize=8)
        self.line_trades, = self.ax2_twin.plot([], [], color='blue', linewidth=2, linestyle='--', label='Trades')
        
        lines = [self.line_winrate, self.line_trades]
        labels = [l.get_label() for l in lines]
        self.ax2.legend(lines, labels, loc='upper left', fontsize=8)
        
        # Setup subplot 3 - Latency
        self.ax3.set_title('Tick Latency (μs)', fontsize=10)
        self.ax3.set_xlabel('Time', fontsize=8)
        self.ax3.set_ylabel('Latency', fontsize=8)
        self.line_latency, = self.ax3.plot([], [], color='red', linewidth=2, label='Latency')
        self.ax3.legend(loc='upper left', fontsize=8)
        
        # Embed matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def start(self):
        if not self.is_running:
            self.is_running = True
            threading.Thread(target=self.update_loop, daemon=True).start()
            
    def stop(self):
        self.is_running = False
        
    def reset(self):
        self.chart_data = {
            'timestamps': deque(maxlen=300),
            'pnl': deque(maxlen=300),
            'trades': deque(maxlen=300),
            'win_rate': deque(maxlen=300),
            'latency': deque(maxlen=300)
        }
        self.line_pnl.set_data([], [])
        self.line_winrate.set_data([], [])
        self.line_trades.set_data([], [])
        self.line_latency.set_data([], [])
        self.canvas.draw()
        
    def update_loop(self):
        """Generate random data and update chart"""
        pnl = 0
        trades = 0
        
        while self.is_running:
            try:
                # Generate random data
                current_time = datetime.now()
                pnl += random.uniform(-10, 15)  # Slight upward bias
                trades += random.randint(0, 3)
                win_rate = random.uniform(45, 75)
                latency = random.uniform(50, 200)
                
                # Store data
                self.chart_data['timestamps'].append(current_time)
                self.chart_data['pnl'].append(pnl)
                self.chart_data['trades'].append(trades)
                self.chart_data['win_rate'].append(win_rate)
                self.chart_data['latency'].append(latency)
                
                # Update chart
                self.update_chart()
                
                time.sleep(0.5)  # Update every 0.5 seconds
                
            except Exception as e:
                print(f"Update error: {e}")
                
    def update_chart(self):
        """Update performance chart with latest data"""
        try:
            if len(self.chart_data['timestamps']) < 2:
                return
            
            timestamps = list(self.chart_data['timestamps'])
            
            # Update PnL chart
            self.line_pnl.set_data(timestamps, list(self.chart_data['pnl']))
            self.ax1.relim()
            self.ax1.autoscale_view()
            
            # Update Win Rate & Trades chart
            self.line_winrate.set_data(timestamps, list(self.chart_data['win_rate']))
            self.line_trades.set_data(timestamps, list(self.chart_data['trades']))
            self.ax2.relim()
            self.ax2.autoscale_view()
            self.ax2_twin.relim()
            self.ax2_twin.autoscale_view()
            
            # Update Latency chart
            self.line_latency.set_data(timestamps, list(self.chart_data['latency']))
            self.ax3.relim()
            self.ax3.autoscale_view()
            
            # Format x-axis with time
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.tick_params(axis='x', rotation=45, labelsize=7)
            
            # Redraw canvas
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Chart update error: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ChartTest(root)
    root.mainloop()
