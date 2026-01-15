"""
Aventa HFT Pro 2026 - Modern GUI Launcher
Professional trading interface with real-time monitoring
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import json
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import sys
import os
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Core modules will be imported lazily on-demand
# This speeds up initial GUI loading significantly

# Translation dictionaries
TRANSLATIONS = {
    'EN': {
        'title': '🚀 AVENTA HFT PRO 2026 v7.0',
        'tab_control': '⚙️ Control Panel',
        'tab_performance': '📊 Performance',
        'tab_risk': '🛡️ Risk Management',
        'tab_ml': '🤖 ML Models',
        'tab_strategy': '🧪 Strategy Tester',
        'tab_logs': '📝 Logs',
        'status_ready': 'Status: Ready',
        'status_trading': 'Status: TRADING ACTIVE',
        'status_stopped': 'Status: Stopped',
        'status_initializing': 'Status: Initializing...',
        
        # Control Panel
        'subtab_config': '⚙️ Configuration',
        'subtab_status': '📊 Status & Logs',
        'config_section': 'Configuration',
        'symbol': 'Symbol:',
        'default_volume': 'Default Volume:',
        'magic_number': 'Magic Number:',
        'risk_per_trade': 'Risk per Trade (%):',
        'min_signal_strength': 'Min Signal Strength:',
        'max_spread': 'Max Spread:',
        'max_volatility': 'Max Volatility:',
        'filling_mode': 'Filling Mode:',
        'sl_multiplier': 'SL Multiplier (ATR):',
        'risk_reward': 'Risk:Reward Ratio:',
        'tp_mode': 'TP Mode:',
        'tp_amount': 'TP Amount ($):',
        'max_floating_loss': 'Max Floating Loss ($):',
        'take_profit_target': 'Take Profit Target ($):',
        'enable_ml': 'Enable ML Predictions',
        'mt5_path': 'MT5 Path:',
        'browse': '📁 Browse',
        'start_trading': '▶️ START TRADING',
        'stop_trading': '⏹️ STOP TRADING',
        'save_config': '💾 Save Config',
        'load_config': '📁 Load Config',
        'quick_load_presets': 'Quick Load Presets',
        'gold_config': '🥇 GOLD Config',
        'eurusd_config': '💱 EURUSD Config',
        'xauusd_config': '🏅 XAUUSD Config',
        'btcusd_config': '₿ BTCUSD Config',
        'current_status': 'Current Status & System Logs',
        'clear_logs': '🗑️ Clear Logs',
        'export_logs': '💾 Export Logs',
        'refresh': '🔄 Refresh',
        
        # Performance Tab
        'performance_metrics': 'Real-Time Performance Metrics',
        'tick_latency': 'Tick Latency (μs):',
        'exec_time': 'Execution Time (ms):',
        'ticks_processed': 'Ticks Processed:',
        'signals_generated': 'Signals Generated:',
        'trades_today': 'Trades Today:',
        'daily_pnl': 'Daily PnL:',
        'win_rate': 'Win Rate:',
        'current_position': 'Current Position:',
        'performance_chart': 'Performance Chart',
        'reset_chart': '🔄 Reset Chart',
        'realtime_updates': '📈 Real-time updates every 1 second',
        
        # Risk Management
        'risk_limits': 'Risk Limits Configuration',
        'max_daily_loss': 'Max Daily Loss ($):',
        'max_daily_trades': 'Max Daily Trades:',
        'max_position_size': 'Max Position Size:',
        'max_positions': 'Max Positions:',
        'max_drawdown': 'Max Drawdown (%):',
        'update_refresh': '🔄 Update & Refresh Status',
        'risk_metrics': 'Risk Metrics & Status',
        'risk_events': 'Risk Events & Alerts',
        
        # ML Models
        'ml_config': 'ML Configuration',
        'training_days': 'Training Days:',
        'train_models': '🎓 Train Models',
        'save_models': '💾 Save Models',
        'load_models': '📁 Load Models',
        'refresh_status': '🔄 Refresh Status',
        'ml_status': 'ML Status & Information',
        'ml_logs': 'ML Training & Prediction Logs',
        
        # Messages
        'system_initialized': 'System initialized. Ready to trade.',
        'system_ready': 'System ready. All components available on-demand.',
        'loading_core': 'Loading core trading modules...',
        'core_loaded': '✓ Core modules loaded',
        'loading_charts': 'Loading chart visualization library...',
        'charts_loaded': '✓ Chart library loaded',
        'initializing_charts': 'Initializing performance charts...',
        'initializing_risk': 'Initializing risk management panel...',
        'initializing_ml': 'Initializing ML panel...',
        'training_days_error': 'Training Days Error',
        'training_days_max': 'Training days cannot exceed 90 days.\nPlease enter a value between 1 and 90.',
        
        # ML Status Panel
        'ml_status_title': '🤖 MACHINE LEARNING STATUS',
        'ml_predictions_label': 'ML Predictions:',
        'enabled': 'ENABLED',
        'disabled': 'DISABLED',
        'no_models_loaded': '⚠️ NO MODELS LOADED',
        'to_use_ml': 'To use ML predictions:',
        'train_new_models': "1. Click '🎓 Train Models' to train new models",
        'or_text': '   OR',
        'load_existing_models': "2. Click '📁 Load Models' to load existing models",
        'training_info_text': 'Training will download historical data and train\n3 neural network models (LSTM, GRU, CNN).',
        'recommended_days': 'Recommended training days: 30-90',
        'training_time': 'Training time: 5-15 minutes depending on data',
        'performance_tips_title': '💡 PERFORMANCE TIPS:',
        'tip_more_data': '• More training data = Better predictions',
        'tip_retrain': '• Retrain monthly for best results',
        'tip_signal_strength': '• ML works best with min_signal_strength: 0.4-0.6',
        'tip_combine': '• Combine ML with technical analysis',
        'tip_monitor': '• Monitor prediction accuracy in logs',
        'model_info_title': '📊 MODEL INFORMATION:',
        'models_loaded': 'Models Loaded: ✓ YES',
        'model_type': 'Model Type: LSTM + GRU + CNN Ensemble',
        'features': 'Features: Price, Volume, Technical Indicators',
        'prediction_window': 'Prediction Window: Next 1-5 ticks',
        'model_arch_title': '🏗️ MODEL ARCHITECTURE:',
        'lstm_desc': '• LSTM (Long Short-Term Memory)',
        'lstm_seq': '  - Sequence length: 50 ticks',
        'lstm_units': '  - Hidden units: 128 → 64',
        'lstm_purpose': '  - Purpose: Temporal patterns',
        'gru_desc': '• GRU (Gated Recurrent Unit)',
        'gru_seq': '  - Sequence length: 50 ticks',
        'gru_units': '  - Hidden units: 128 → 64',
        'gru_purpose': '  - Purpose: Fast pattern recognition',
        'cnn_desc': '• CNN (Convolutional Neural Network)',
        'cnn_filters': '  - Filters: 64 → 128 → 64',
        'cnn_kernel': '  - Kernel size: 3',
        'cnn_purpose': '  - Purpose: Local pattern detection',
        'ensemble_desc': '• Ensemble Voting',
        'ensemble_combines': '  - Combines 3 models',
        'ensemble_threshold': '  - Confidence threshold: 60%',
        'training_info_title': '📈 TRAINING INFORMATION:',
        'current_symbol': 'Current Symbol:',
        'model_location': 'Model Location:',
        'available_models': 'Available Models:',
        'last_trained': 'Last Trained:',
        'status_ready': 'Status: ✓ Ready for predictions',
        'no_trained_models': '⚠️ No trained models for',
        'expected_location': 'Expected location:',
        'all_available_models': '📁 ALL AVAILABLE MODELS:',
        'use_load_models': "Use '📁 Load Models' to switch between symbols",
        'usage_guide_title': '📖 USAGE GUIDE:',
        'usage_train': '1. Train Models:',
        'usage_train_1': '   - Set training days (30-90 recommended)',
        'usage_train_2': "   - Click '🎓 Train Models'",
        'usage_train_3': '   - Wait for training to complete',
        'usage_use': '2. Use Predictions:',
        'usage_use_1': '   - Enable ML checkbox in Control Panel',
        'usage_use_2': '   - Start trading normally',
        'usage_use_3': '   - ML predictions enhance signals',
        'usage_manage': '3. Model Management:',
        'usage_manage_1': '   - Save: Store trained models',
        'usage_manage_2': '   - Load: Restore saved models',
        'usage_manage_3': '   - Retrain: Update with new data',
        'ml_initialized': 'ML module initialized. Ready to train or load models.',
    },
    'ID': {
        'title': '🚀 AVENTA HFT PRO 2026',
        'tab_control': '⚙️ Panel Kontrol',
        'tab_performance': '📊 Performa',
        'tab_risk': '🛡️ Manajemen Risiko',
        'tab_ml': '🤖 Model ML',
        'tab_strategy': '🧪 Penguji Strategi',
        'tab_logs': '📝 Log',
        'status_ready': 'Status: Siap',
        'status_trading': 'Status: TRADING AKTIF',
        'status_stopped': 'Status: Dihentikan',
        'status_initializing': 'Status: Menginisialisasi...',
        
        # Control Panel
        'subtab_config': '⚙️ Konfigurasi',
        'subtab_status': '📊 Status & Log',
        'config_section': 'Konfigurasi',
        'symbol': 'Simbol:',
        'default_volume': 'Volume Default:',
        'magic_number': 'Magic Number:',
        'risk_per_trade': 'Risiko per Trade (%):',
        'min_signal_strength': 'Kekuatan Sinyal Min:',
        'max_spread': 'Spread Maks:',
        'max_volatility': 'Volatilitas Maks:',
        'filling_mode': 'Mode Pengisian:',
        'sl_multiplier': 'Pengali SL (ATR):',
        'risk_reward': 'Rasio Risk:Reward:',
        'tp_mode': 'Mode TP:',
        'tp_amount': 'Jumlah TP ($):',
        'max_floating_loss': 'Floating Loss Maks ($):',
        'take_profit_target': 'Target Take Profit ($):',
        'enable_ml': 'Aktifkan Prediksi ML',
        'mt5_path': 'Path MT5:',
        'browse': '📁 Telusuri',
        'start_trading': '▶️ MULAI TRADING',
        'stop_trading': '⏹️ STOP TRADING',
        'save_config': '💾 Simpan Konfigurasi',
        'load_config': '📁 Muat Konfigurasi',
        'quick_load_presets': 'Muat Preset Cepat',
        'gold_config': '🥇 Config GOLD',
        'eurusd_config': '💱 Config EURUSD',
        'xauusd_config': '🏅 Config XAUUSD',
        'btcusd_config': '₿ Config BTCUSD',
        'current_status': 'Status & Log Sistem Saat Ini',
        'clear_logs': '🗑️ Hapus Log',
        'export_logs': '💾 Ekspor Log',
        'refresh': '🔄 Segarkan',
        
        # Performance Tab
        'performance_metrics': 'Metrik Performa Real-Time',
        'tick_latency': 'Latensi Tick (μs):',
        'exec_time': 'Waktu Eksekusi (ms):',
        'ticks_processed': 'Tick Diproses:',
        'signals_generated': 'Sinyal Dihasilkan:',
        'trades_today': 'Trade Hari Ini:',
        'daily_pnl': 'PnL Harian:',
        'win_rate': 'Win Rate:',
        'current_position': 'Posisi Saat Ini:',
        'performance_chart': 'Grafik Performa',
        'reset_chart': '🔄 Reset Grafik',
        'realtime_updates': '📈 Update real-time setiap 1 detik',
        
        # Risk Management
        'risk_limits': 'Konfigurasi Batas Risiko',
        'max_daily_loss': 'Loss Harian Maks ($):',
        'max_daily_trades': 'Trade Harian Maks:',
        'max_position_size': 'Ukuran Posisi Maks:',
        'max_positions': 'Posisi Maks:',
        'max_drawdown': 'Drawdown Maks (%):',
        'update_refresh': '🔄 Update & Segarkan Status',
        'risk_metrics': 'Metrik & Status Risiko',
        'risk_events': 'Event & Alert Risiko',
        
        # ML Models
        'ml_config': 'Konfigurasi ML',
        'training_days': 'Hari Training:',
        'train_models': '🎓 Train Model',
        'save_models': '💾 Simpan Model',
        'load_models': '📁 Muat Model',
        'refresh_status': '🔄 Segarkan Status',
        'ml_status': 'Status & Informasi ML',
        'ml_logs': 'Log Training & Prediksi ML',
        
        # Messages
        'system_initialized': 'Sistem diinisialisasi. Siap untuk trading.',
        'system_ready': 'Sistem siap. Semua komponen tersedia sesuai kebutuhan.',
        'loading_core': 'Memuat modul trading inti...',
        'core_loaded': '✓ Modul inti dimuat',
        'loading_charts': 'Memuat library visualisasi grafik...',
        'charts_loaded': '✓ Library grafik dimuat',
        'initializing_charts': 'Menginisialisasi grafik performa...',
        'initializing_risk': 'Menginisialisasi panel manajemen risiko...',
        'initializing_ml': 'Menginisialisasi panel ML...',
        'training_days_error': 'Error Training Days',
        'training_days_max': 'Training days tidak boleh lebih dari 90 hari.\\nMohon masukkan nilai antara 1 hingga 90.',
        
        # ML Status Panel
        'ml_status_title': '🤖 STATUS MACHINE LEARNING',
        'ml_predictions_label': 'Prediksi ML:',
        'enabled': 'AKTIF',
        'disabled': 'NONAKTIF',
        'no_models_loaded': '⚠️ TIDAK ADA MODEL DIMUAT',
        'to_use_ml': 'Untuk menggunakan prediksi ML:',
        'train_new_models': "1. Klik '🎓 Train Models' untuk melatih model baru",
        'or_text': '   ATAU',
        'load_existing_models': "2. Klik '📁 Load Models' untuk memuat model yang ada",
        'training_info_text': 'Training akan mengunduh data historis dan melatih\\n3 model neural network (LSTM, GRU, CNN).',
        'recommended_days': 'Training days yang direkomendasikan: 30-90',
        'training_time': 'Waktu training: 5-15 menit tergantung data',
        'performance_tips_title': '💡 TIPS PERFORMA:',
        'tip_more_data': '• Lebih banyak data training = Prediksi lebih baik',
        'tip_retrain': '• Latih ulang setiap bulan untuk hasil terbaik',
        'tip_signal_strength': '• ML bekerja optimal dengan min_signal_strength: 0.4-0.6',
        'tip_combine': '• Kombinasikan ML dengan analisa teknikal',
        'tip_monitor': '• Pantau akurasi prediksi di logs',
        'model_info_title': '📊 INFORMASI MODEL:',
        'models_loaded': 'Model Dimuat: ✓ YA',
        'model_type': 'Tipe Model: LSTM + GRU + CNN Ensemble',
        'features': 'Fitur: Harga, Volume, Indikator Teknikal',
        'prediction_window': 'Jendela Prediksi: 1-5 tick berikutnya',
        'model_arch_title': '🏗️ ARSITEKTUR MODEL:',
        'lstm_desc': '• LSTM (Long Short-Term Memory)',
        'lstm_seq': '  - Panjang sequence: 50 tick',
        'lstm_units': '  - Hidden units: 128 → 64',
        'lstm_purpose': '  - Tujuan: Pola temporal',
        'gru_desc': '• GRU (Gated Recurrent Unit)',
        'gru_seq': '  - Panjang sequence: 50 tick',
        'gru_units': '  - Hidden units: 128 → 64',
        'gru_purpose': '  - Tujuan: Pengenalan pola cepat',
        'cnn_desc': '• CNN (Convolutional Neural Network)',
        'cnn_filters': '  - Filter: 64 → 128 → 64',
        'cnn_kernel': '  - Ukuran kernel: 3',
        'cnn_purpose': '  - Tujuan: Deteksi pola lokal',
        'ensemble_desc': '• Ensemble Voting',
        'ensemble_combines': '  - Menggabungkan 3 model',
        'ensemble_threshold': '  - Threshold confidence: 60%',
        'training_info_title': '📈 INFORMASI TRAINING:',
        'current_symbol': 'Symbol Saat Ini:',
        'model_location': 'Lokasi Model:',
        'available_models': 'Model Tersedia:',
        'last_trained': 'Terakhir Dilatih:',
        'status_ready': 'Status: ✓ Siap untuk prediksi',
        'no_trained_models': '⚠️ Tidak ada model terlatih untuk',
        'expected_location': 'Lokasi yang diharapkan:',
        'all_available_models': '📁 SEMUA MODEL TERSEDIA:',
        'use_load_models': "Gunakan '📁 Load Models' untuk berganti antar symbol",
        'usage_guide_title': '📖 PANDUAN PENGGUNAAN:',
        'usage_train': '1. Latih Model:',
        'usage_train_1': '   - Set training days (30-90 direkomendasikan)',
        'usage_train_2': "   - Klik '🎓 Train Models'",
        'usage_train_3': '   - Tunggu hingga training selesai',
        'usage_use': '2. Gunakan Prediksi:',
        'usage_use_1': '   - Aktifkan checkbox ML di Panel Kontrol',
        'usage_use_2': '   - Mulai trading seperti biasa',
        'usage_use_3': '   - Prediksi ML meningkatkan sinyal',
        'usage_manage': '3. Manajemen Model:',
        'usage_manage_1': '   - Save: Simpan model terlatih',
        'usage_manage_2': '   - Load: Pulihkan model tersimpan',
        'usage_manage_3': '   - Retrain: Update dengan data baru',
        'ml_initialized': 'Modul ML diinisialisasi. Siap untuk melatih atau memuat model.',
    }
}


class HFTProGUI:
    def on_language_changed(self, event=None):
        self.refresh_language()
        lang_name = "English" if self.current_language.get() == 'EN' else "Bahasa Indonesia"
        self.log_message(f"Language changed to {lang_name}", "INFO")
    """Modern GUI for Aventa HFT Pro 2026"""

    def __init__(self, root):
        self.root = root
        self.root.title("Aventa HFT Pro 2026 - Ultra Low Latency Trading System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0e27')
        
        # Language setting
        self.current_language = tk.StringVar(value='EN')
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        # Trading components (lazy loaded)
        self.engine = None
        self.risk_manager = None
        self.ml_predictor = None
        self.config = {}
        
        # Lazy loading flags
        self.chart_initialized = False
        self.matplotlib_loaded = False
        self.core_modules_loaded = False
        
        # State
        self.is_running = False
        self.update_thread = None
        
        # Performance data for charts (will be populated when chart tab is opened)
        self.chart_data = {
            'timestamps': deque(maxlen=300),  # Last 5 minutes at 1s update
            'pnl': deque(maxlen=300),
            'equity': deque(maxlen=300),
            'trades': deque(maxlen=300),
            'win_rate': deque(maxlen=300),
            'latency': deque(maxlen=300)
        }
        
        # Initialize variables that are needed for config loading
        # (before tabs are built)
        self.limit_vars = {
            'max_daily_loss': tk.StringVar(value="1000"),
            'max_daily_trades': tk.StringVar(value="500"),
            'max_position_size': tk.StringVar(value="1.0"),
            'max_positions': tk.StringVar(value="3"),
            'max_drawdown_pct': tk.StringVar(value="10")
        }
        
        # ML configuration variable
        self.ml_days_var = tk.StringVar(value="40")
        
        # Create GUI (fast, without heavy components)
        self.create_gui()
        
        # Load configuration asynchronously
        self.root.after(100, self.async_init)
    
    def t(self, key):
        """Get translated text for current language"""
        lang = self.current_language.get()
        return TRANSLATIONS.get(lang, TRANSLATIONS['EN']).get(key, key)
    
    # Run backtest in thread to avoid blocking UI
    def backtest_thread(self):
        try:
            # ...existing code to fetch data...
            # Simulasi backtest dengan parameter risk/money management
            try:
                initial_balance = float(self.test_initial_balance.get())
            except Exception:
                initial_balance = 10000.0  # fallback default
            balance = initial_balance
            equity = initial_balance
            floating_pnl = 0.0
            max_equity = initial_balance
            min_equity = initial_balance
            daily_loss = 0.0
            daily_trades = 0
            open_positions = []
            trade_history = []
            drawdown = 0.0
            # ...existing code to fetch price data, signals, etc...
            # (Pseudo code, replace with your actual backtest loop)
            pass  # Ganti dengan implementasi backtest Anda
        except Exception as e:
            self.update_backtest_status(f"Backtest error: {e}", error=True)

    def run_backtest(self):
        import threading
        threading.Thread(target=self.backtest_thread, daemon=True).start()
    
    def refresh_language(self):
        """Refresh all GUI text with current language"""
        try:
            # Update title
            if hasattr(self, 'title_label'):
                self.title_label.config(text=self.t('title'))
            
            # Update status bar
            if hasattr(self, 'status_bar'):
                current_status = self.status_bar.cget('text')
                if 'Ready' in current_status or 'Siap' in current_status:
                    self.status_bar.config(text=self.t('status_ready'))
                elif 'TRADING' in current_status:
                    self.status_bar.config(text=self.t('status_trading'))
                elif 'Stopped' in current_status or 'Dihentikan' in current_status:
                    self.status_bar.config(text=self.t('status_stopped'))
            
            # Update notebook tabs
            if hasattr(self, 'notebook'):
                self.notebook.tab(self.control_tab, text=self.t('tab_control'))
                self.notebook.tab(self.performance_tab, text=self.t('tab_performance'))
                self.notebook.tab(self.risk_tab, text=self.t('tab_risk'))
                self.notebook.tab(self.ml_tab, text=self.t('tab_ml'))
                self.notebook.tab(self.log_tab, text=self.t('tab_logs'))
            
            # Rebuild current tab to update all labels
            current_tab = self.notebook.select()
            tab_text = self.notebook.tab(current_tab, "text")
            
            # Rebuild tabs that are already built
            if (self.t('tab_control') in tab_text or 'Panel Kontrol' in tab_text) and hasattr(self, 'start_btn'):
                self.rebuild_control_tab()
            elif (self.t('tab_performance') in tab_text or 'Performa' in tab_text) and self.chart_initialized:
                self.rebuild_performance_tab()
            elif (self.t('tab_risk') in tab_text or 'Risiko' in tab_text) and hasattr(self, 'risk_metrics_text'):
                self.rebuild_risk_tab()
            elif (self.t('tab_ml') in tab_text or 'Model ML' in tab_text) and hasattr(self, 'ml_info_text'):
                self.rebuild_ml_tab()
            elif (self.t('tab_logs') in tab_text or 'Log' in tab_text) and hasattr(self, 'log_text'):
                self.rebuild_log_tab()
                
        except Exception as e:
            print(f"Error refreshing language: {e}")
        
    def async_init(self):
        """Initialize heavy components asynchronously after GUI is displayed"""
        self.status_bar.config(text="Status: Initializing...")
        
        def init_thread():
            try:
                # Load configuration
                self.load_config()
                
                # Update status
                self.root.after(0, lambda: self.status_bar.config(text=self.t('status_ready')))
                self.root.after(0, lambda: self.log_message(self.t('system_ready'), "SUCCESS"))
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"Initialization warning: {str(e)}", "WARNING"))
        
        # Run in background thread
        threading.Thread(target=init_thread, daemon=True).start()
    
    def load_core_modules(self):
        """Lazy load core trading modules"""
        if self.core_modules_loaded:
            return
        
        try:
            self.log_message(self.t('loading_core'), "INFO")
            
            # Import heavy modules only when needed
            global UltraLowLatencyEngine, RiskManager, MLPredictor
            from aventa_hft_core import UltraLowLatencyEngine
            from risk_manager import RiskManager
            from ml_predictor import MLPredictor
            
            self.core_modules_loaded = True
            self.log_message(self.t('core_loaded'), "SUCCESS")
        except Exception as e:
            self.log_message(f"Error loading core modules: {str(e)}", "ERROR")
            raise
    
    def load_matplotlib(self):
        """Lazy load matplotlib for charts"""
        if self.matplotlib_loaded:
            return
        
        try:
            self.log_message(self.t('loading_charts'), "INFO")
            
            # Import matplotlib only when chart tab is opened
            global plt, FigureCanvasTkAgg, Figure, mdates
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            import matplotlib.dates as mdates
            
            self.matplotlib_loaded = True
            self.log_message(self.t('charts_loaded'), "SUCCESS")
        except Exception as e:
            self.log_message(f"Error loading chart library: {str(e)}", "ERROR")
            raise
    
    def configure_styles(self):
        """Configure modern dark theme styles"""
        # Configure colors - Modern Cool Theme
        bg_color = '#0a0e27'           # Deep navy background
        label_bg = "#DDDEE0"         # Light gray for labels
        secondary_bg = "#DDDEE0"        # Light gray for entries
        tertiary_bg = "#ffffff"         # White for buttons
        fg_color = '#0a0e27'            # Dark text
        accent_color = "#0C0C0C"        # Dark accent
        accent_secondary = '#7c4dff'    # Purple accent
        success_color = '#00e676'       # Bright green
        danger_color = '#ff1744'        # Bright red
        warning_color = '#ffd600'       # Bright yellow
        info_color = '#00b0ff'          # Bright blue
        
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabel', background=label_bg, foreground=fg_color, font=('Segoe UI', 10))
        self.style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'), foreground='#ffffff', background=bg_color)
        self.style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'), foreground=accent_secondary)
        self.style.configure('Value.TLabel', font=('Segoe UI', 11, 'bold'), foreground=success_color)
        self.style.configure('Danger.TLabel', font=('Segoe UI', 11, 'bold'), foreground=danger_color)
        
        self.style.configure('TButton', font=('Segoe UI', 10, 'bold'), borderwidth=0, relief='flat')
        self.style.map('TButton', background=[('active', accent_secondary), ('!active', tertiary_bg)])
        
        self.style.configure('Start.TButton', background=success_color, foreground='#000000')
        self.style.configure('Stop.TButton', background=danger_color, foreground='#ffffff')
        
        self.style.configure('TEntry', fieldbackground=secondary_bg, foreground=accent_color, borderwidth=1, relief='flat')
        self.style.configure('TCombobox', fieldbackground=secondary_bg, foreground=accent_color, borderwidth=1)
    
    def create_gui(self):
        """Create main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header frame for title and language selector
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title (left side)
        self.title_label = ttk.Label(header_frame, text=self.t('title'), style='Title.TLabel')
        self.title_label.pack(side=tk.LEFT)
        
        # Language selector (right side)
        lang_frame = ttk.Frame(header_frame)
        lang_frame.pack(side=tk.RIGHT)
        
        ttk.Label(lang_frame, text="🌐", font=('Segoe UI', 14), 
                 background='#0a0e27', foreground='#ffffff').pack(side=tk.LEFT, padx=(0, 5))
        
        lang_selector = ttk.Combobox(lang_frame, textvariable=self.current_language,
                                     values=['EN', 'ID'], width=5, state='readonly',
                                     font=('Segoe UI', 10, 'bold'))
        lang_selector.pack(side=tk.LEFT)
        lang_selector.bind('<<ComboboxSelected>>', self.on_language_changed)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tabs
        self.control_tab = ttk.Frame(self.notebook)
        self.performance_tab = ttk.Frame(self.notebook)
        self.risk_tab = ttk.Frame(self.notebook)
        self.ml_tab = ttk.Frame(self.notebook)
        self.strategy_tab = ttk.Frame(self.notebook)
        self.log_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.control_tab, text=self.t('tab_control'))
        self.notebook.add(self.performance_tab, text=self.t('tab_performance'))
        self.notebook.add(self.risk_tab, text=self.t('tab_risk'))
        self.notebook.add(self.ml_tab, text=self.t('tab_ml'))
        self.notebook.add(self.strategy_tab, text=self.t('tab_strategy'))
        self.notebook.add(self.log_tab, text=self.t('tab_logs'))
        
        # Bind tab change event for lazy loading
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_changed)
        
        # Build only essential tabs immediately (Control and Logs)
        self.build_control_tab()
        self.build_log_tab()
        
        # Other tabs will be built on-demand when first accessed
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text=self.t('status_ready'), relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
    
    def on_tab_changed(self, event):
        """Handle tab change - lazy load tab content"""
        try:
            current_tab = self.notebook.select()
            tab_text = self.notebook.tab(current_tab, "text")
            
            # Lazy build performance tab (with charts)
            if "Performance" in tab_text or "Performa" in tab_text:
                if not self.chart_initialized:
                    self.log_message(self.t('initializing_charts'), "INFO")
                    self.load_matplotlib()  # Load matplotlib first
                    self.build_performance_tab()  # Build the tab
                    self.chart_initialized = True
            
            # Lazy build risk tab
            elif "Risk" in tab_text or "Risiko" in tab_text:
                if not hasattr(self, 'risk_metrics_text'):
                    self.log_message(self.t('initializing_risk'), "INFO")
                    self.build_risk_tab()
            
            # Lazy build ML tab
            elif "ML" in tab_text or "Model ML" in tab_text:
                if not hasattr(self, 'ml_info_text'):
                    self.log_message(self.t('initializing_ml'), "INFO")
                    self.build_ml_tab()
            
            # Lazy build Strategy Tester tab
            elif "Strategy" in tab_text or "Strategi" in tab_text or "Penguji" in tab_text:
                if not hasattr(self, 'backtest_stats_text'):
                    self.log_message("Initializing Strategy Tester...", "INFO")
                    self.build_strategy_tab()
                
        except Exception as e:
            self.log_message(f"Error loading tab: {str(e)}", "ERROR")
    
    def build_control_tab(self):
        """Build control panel tab"""
        # Create sub-notebook inside control tab
        control_notebook = ttk.Notebook(self.control_tab)
        control_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create sub-tabs
        config_subtab = ttk.Frame(control_notebook)
        status_subtab = ttk.Frame(control_notebook)
        
        control_notebook.add(config_subtab, text="⚙️ Configuration")
        control_notebook.add(status_subtab, text="📊 Status & Logs")
        
        # Build configuration sub-tab
        self.build_config_subtab(config_subtab)
        
        # Build status sub-tab
        self.build_status_subtab(status_subtab)
    
    def build_config_subtab(self, parent):
        """Build configuration sub-tab"""
        frame = ttk.Frame(parent, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Configuration section
        self.config_labelframe = ttk.LabelFrame(frame, text=self.t('config_section'), padding="10")
        self.config_labelframe.pack(fill=tk.X, pady=(0, 10))
        
        # Store label references for language updates
        self.config_labels = {}
        
        # Symbol
        self.config_labels['symbol'] = ttk.Label(self.config_labelframe, text=self.t('symbol'))
        self.config_labels['symbol'].grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.symbol_var = tk.StringVar(value="GOLD.ls")
        symbol_entry = ttk.Entry(self.config_labelframe, textvariable=self.symbol_var, width=15)
        symbol_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Volume
        self.config_labels['volume'] = ttk.Label(self.config_labelframe, text=self.t('default_volume'))
        self.config_labels['volume'].grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.volume_var = tk.StringVar(value="0.01")
        volume_entry = ttk.Entry(self.config_labelframe, textvariable=self.volume_var, width=10)
        volume_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Magic Number
        self.config_labels['magic'] = ttk.Label(self.config_labelframe, text=self.t('magic_number'))
        self.config_labels['magic'].grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.magic_var = tk.StringVar(value="2026001")
        magic_entry = ttk.Entry(self.config_labelframe, textvariable=self.magic_var, width=15)
        magic_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Risk per trade
        self.config_labels['risk'] = ttk.Label(self.config_labelframe, text=self.t('risk_per_trade'))
        self.config_labels['risk'].grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.risk_var = tk.StringVar(value="1.0")
        risk_entry = ttk.Entry(self.config_labelframe, textvariable=self.risk_var, width=10)
        risk_entry.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Min signal strength
        self.config_labels['signal'] = ttk.Label(self.config_labelframe, text=self.t('min_signal_strength'))
        self.config_labels['signal'].grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.signal_strength_var = tk.StringVar(value="0.4")
        signal_entry = ttk.Entry(self.config_labelframe, textvariable=self.signal_strength_var, width=10)
        signal_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.config_labelframe, text="(0.3-0.5 = more trades | 0.6-0.8 = fewer, safer)", 
                 font=('Segoe UI', 8)).grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=5)
        
        # Max Spread
        self.config_labels['spread'] = ttk.Label(self.config_labelframe, text=self.t('max_spread'))
        self.config_labels['spread'].grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_spread_var = tk.StringVar(value="0.05")
        spread_entry = ttk.Entry(self.config_labelframe, textvariable=self.max_spread_var, width=10)
        spread_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.config_labelframe, text="(GOLD: 0.02-0.05 | EURUSD: 0.0002-0.001)", 
                 font=('Segoe UI', 8)).grid(row=3, column=2, columnspan=2, sticky=tk.W, padx=5)
        
        # Max Volatility
        self.config_labels['volatility'] = ttk.Label(self.config_labelframe, text=self.t('max_volatility'))
        self.config_labels['volatility'].grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_volatility_var = tk.StringVar(value="0.005")
        volatility_entry = ttk.Entry(self.config_labelframe, textvariable=self.max_volatility_var, width=10)
        volatility_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.config_labelframe, text="(Higher = allow more volatile markets)", 
                 font=('Segoe UI', 8)).grid(row=4, column=2, columnspan=2, sticky=tk.W, padx=5)
        
        # Filling Mode
        self.config_labels['filling'] = ttk.Label(self.config_labelframe, text=self.t('filling_mode'))
        self.config_labels['filling'].grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.filling_mode_var = tk.StringVar(value="FOK")
        filling_combo = ttk.Combobox(self.config_labelframe, textvariable=self.filling_mode_var, 
                                     values=["FOK", "IOC", "RETURN"], width=12, state="readonly")
        filling_combo.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.config_labelframe, text="(FOK=Fill or Kill | IOC=Immediate or Cancel | RETURN=Market)", 
                 font=('Segoe UI', 8)).grid(row=5, column=2, columnspan=2, sticky=tk.W, padx=5)
        
        # SL Multiplier
        self.config_labels['sl'] = ttk.Label(self.config_labelframe, text=self.t('sl_multiplier'))
        self.config_labels['sl'].grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.sl_multiplier_var = tk.StringVar(value="2.0")
        sl_entry = ttk.Entry(self.config_labelframe, textvariable=self.sl_multiplier_var, width=10)
        sl_entry.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.config_labelframe, text="(SL = ATR × this value | 1.5-2.5 typical)", 
                 font=('Segoe UI', 8)).grid(row=6, column=2, columnspan=2, sticky=tk.W, padx=5)
        
        # Risk:Reward Ratio
        self.config_labels['rr'] = ttk.Label(self.config_labelframe, text=self.t('risk_reward'))
        self.config_labels['rr'].grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.risk_reward_var = tk.StringVar(value="2.0")
        rr_entry = ttk.Entry(self.config_labelframe, textvariable=self.risk_reward_var, width=10)
        rr_entry.grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.config_labelframe, text="(TP = SL × this value | 1.5-3.0 typical)", 
                 font=('Segoe UI', 8)).grid(row=7, column=2, columnspan=2, sticky=tk.W, padx=5)
        
        # TP Mode and Dollar Amount
        self.config_labels['tp_mode'] = ttk.Label(self.config_labelframe, text=self.t('tp_mode'))
        self.config_labels['tp_mode'].grid(row=7, column=4, sticky=tk.W, padx=5, pady=5)
        self.tp_mode_var = tk.StringVar(value="RiskReward")
        tp_mode_combo = ttk.Combobox(self.config_labelframe, textvariable=self.tp_mode_var, 
                                     values=["RiskReward", "FixedDollar"], width=12, state="readonly")
        tp_mode_combo.grid(row=7, column=5, sticky=tk.W, padx=5, pady=5)
        
        self.config_labels['tp_amt'] = ttk.Label(self.config_labelframe, text=self.t('tp_amount'))
        self.config_labels['tp_amt'].grid(row=7, column=6, sticky=tk.W, padx=5, pady=5)
        self.tp_dollar_var = tk.StringVar(value="0.5")
        tp_dollar_entry = ttk.Entry(self.config_labelframe, textvariable=self.tp_dollar_var, width=8)
        tp_dollar_entry.grid(row=7, column=7, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.config_labelframe, text="(if FixedDollar mode)", 
                 font=('Segoe UI', 8)).grid(row=7, column=8, sticky=tk.W, padx=5)
        
        # Max Floating Loss
        self.config_labels['floating_loss'] = ttk.Label(self.config_labelframe, text=self.t('max_floating_loss'))
        self.config_labels['floating_loss'].grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_floating_loss_var = tk.StringVar(value="500")
        mfl_entry = ttk.Entry(self.config_labelframe, textvariable=self.max_floating_loss_var, width=10)
        mfl_entry.grid(row=8, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.config_labelframe, text="(Stop opening new positions if floating loss exceeds this)", 
                 font=('Segoe UI', 8)).grid(row=8, column=2, columnspan=2, sticky=tk.W, padx=5)
        
        # Max Floating Profit (Close All Target)
        self.config_labels['tp_target'] = ttk.Label(self.config_labelframe, text=self.t('take_profit_target'))
        self.config_labels['tp_target'].grid(row=8, column=4, sticky=tk.W, padx=5, pady=5)
        self.max_floating_profit_var = tk.StringVar(value="1")
        mfp_entry = ttk.Entry(self.config_labelframe, textvariable=self.max_floating_profit_var, width=10)
        mfp_entry.grid(row=8, column=5, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.config_labelframe, text="(Close all positions when profit reaches this)", 
                 font=('Segoe UI', 8)).grid(row=8, column=6, columnspan=2, sticky=tk.W, padx=5)
        
        # Use ML
        self.use_ml_var = tk.BooleanVar(value=False)
        self.ml_checkbox = ttk.Checkbutton(self.config_labelframe, text=self.t('enable_ml'), variable=self.use_ml_var)
        self.ml_checkbox.grid(row=9, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # MT5 Path
        self.config_labels['mt5'] = ttk.Label(self.config_labelframe, text=self.t('mt5_path'))
        self.config_labels['mt5'].grid(row=10, column=0, sticky=tk.W, padx=5, pady=5)
        self.mt5_path_var = tk.StringVar(value="C:\\Program Files\\XM Global MT5\\terminal64.exe")
        mt5_entry = ttk.Entry(self.config_labelframe, textvariable=self.mt5_path_var, width=60)
        mt5_entry.grid(row=10, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(self.config_labelframe, text=self.t('browse'), command=self.browse_mt5_path, width=10).grid(row=10, column=4, padx=5, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(button_frame, text="▶️ START TRADING", 
                                    command=self.start_trading, style='Start.TButton', width=20)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹️ STOP TRADING", 
                                   command=self.stop_trading, style='Stop.TButton', width=20, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="💾 Save Config", 
                  command=self.save_config, width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="📁 Load Config", 
                  command=self.load_config, width=15).pack(side=tk.LEFT, padx=5)
        
        # Quick load preset buttons
        preset_frame = ttk.LabelFrame(frame, text="Quick Load Presets", padding="7")
        preset_frame.pack(fill=tk.X, pady=(0, 7))
        
        ttk.Button(preset_frame, text="🥇 GOLD Config", 
                  command=lambda: self.quick_load_config("config_GOLD.json"), width=19).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(preset_frame, text="💱 EURUSD Config", 
                  command=lambda: self.quick_load_config("config_EURUSD.json"), width=19).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(preset_frame, text="🏅 XAUUSD Config", 
                  command=lambda: self.quick_load_config("config_XAUUSD.json"), width=19).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(preset_frame, text="₿ BTCUSD Config", 
                  command=lambda: self.quick_load_config("config_BTCUSD.json"), width=19).pack(side=tk.LEFT, padx=5)
    
    def build_status_subtab(self, parent):
        """Build status and logs sub-tab"""
        frame = ttk.Frame(parent, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Current status
        status_frame = ttk.LabelFrame(frame, text="Current Status & System Logs", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=25, 
                                                     bg='#151932', fg='#e8eaf6', 
                                                     font=('Consolas', 10), insertbackground='#00d4ff')
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons at bottom
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="🗑️ Clear Logs", 
                  command=lambda: self.status_text.delete(1.0, tk.END), width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="💾 Export Logs", 
                  command=self.export_status_logs, width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="🔄 Refresh", 
                  command=lambda: self.log_message("Status refreshed", "INFO"), width=15).pack(side=tk.LEFT, padx=5)
        
        # Initialize with welcome message
        self.log_message(self.t('system_initialized'), "INFO")
    
    def rebuild_control_tab(self):
        """Rebuild control tab with current language"""
        # Update buttons
        if hasattr(self, 'start_btn'):
            self.start_btn.config(text=self.t('start_trading'))
        if hasattr(self, 'stop_btn'):
            self.stop_btn.config(text=self.t('stop_trading'))
        
        # Update config labels if they exist
        if hasattr(self, 'config_labels'):
            label_keys = {
                'symbol': 'symbol',
                'volume': 'default_volume',
                'magic': 'magic_number',
                'risk': 'risk_per_trade',
                'signal': 'min_signal_strength',
                'spread': 'max_spread',
                'volatility': 'max_volatility',
                'filling': 'filling_mode',
                'sl': 'sl_multiplier',
                'rr': 'risk_reward',
                'tp_mode': 'tp_mode',
                'tp_amt': 'tp_amount',
                'floating_loss': 'max_floating_loss',
                'tp_target': 'take_profit_target',
                'mt5': 'mt5_path',
            }
            for key, trans_key in label_keys.items():
                if key in self.config_labels:
                    self.config_labels[key].config(text=self.t(trans_key))
        
        # Update config frame title
        if hasattr(self, 'config_labelframe'):
            self.config_labelframe.config(text=self.t('config_section'))
        
        # Update preset frame title  
        if hasattr(self, 'preset_labelframe'):
            self.preset_labelframe.config(text=self.t('quick_load_presets'))
        
        # Update ML checkbox
        if hasattr(self, 'ml_checkbox'):
            self.ml_checkbox.config(text=self.t('enable_ml'))
    
    def rebuild_performance_tab(self):
        """Rebuild performance tab with current language"""
        pass  # Complex rebuild not needed for now
    
    def rebuild_risk_tab(self):
        """Rebuild risk tab with current language"""
        pass  # Complex rebuild not needed for now
    
    def rebuild_ml_tab(self):
        """Rebuild ML tab with current language"""
        pass  # Complex rebuild not needed for now
    
    def rebuild_log_tab(self):
        """Rebuild log tab with current language"""
        pass  # Complex rebuild not needed for now
    
    def build_performance_tab(self):
        """Build performance monitoring tab"""
        # Clear any existing content
        for widget in self.performance_tab.winfo_children():
            widget.destroy()
        
        frame = ttk.Frame(self.performance_tab, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Real-time metrics
        metrics_frame = ttk.LabelFrame(frame, text="Real-Time Performance Metrics", padding="10")
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create metric labels
        metrics = [
            ("Tick Latency (μs):", "tick_latency"),
            ("Execution Time (ms):", "exec_time"),
            ("Ticks Processed:", "ticks_processed"),
            ("Signals Generated:", "signals_generated"),
            ("Trades Today:", "trades_today"),
            ("Daily PnL:", "daily_pnl"),
            ("Win Rate:", "win_rate"),
            ("Current Position:", "current_position"),
        ]
        
        self.metric_labels = {}
        for idx, (label, key) in enumerate(metrics):
            row = idx // 2
            col = (idx % 2) * 2
            
            ttk.Label(metrics_frame, text=label, style='Header.TLabel').grid(
                row=row, column=col, sticky=tk.W, padx=10, pady=5
            )
            
            value_label = ttk.Label(metrics_frame, text="--", style='Value.TLabel')
            value_label.grid(row=row, column=col+1, sticky=tk.W, padx=10, pady=5)
            self.metric_labels[key] = value_label
        
        # Chart frame
        chart_frame = ttk.LabelFrame(frame, text="Performance Chart", padding="10")
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure with subplots
        self.fig = Figure(figsize=(10, 6), facecolor='#0a0e27')
        self.fig.subplots_adjust(hspace=0.4)
        
        # Create 3 subplots
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        
        # Configure subplot styles
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor('#151932')
            ax.tick_params(colors='#e8eaf6', labelsize=8)
            ax.spines['bottom'].set_color('#7c4dff')
            ax.spines['top'].set_color('#7c4dff')
            ax.spines['left'].set_color('#7c4dff')
            ax.spines['right'].set_color('#7c4dff')
            ax.grid(True, alpha=0.3, color='#00d4ff', linestyle='--')
        
        # Setup subplot 1 - PnL
        self.ax1.set_title('Daily PnL ($)', color='#00d4ff', fontsize=10, pad=5, weight='bold')
        self.ax1.set_ylabel('PnL', color='#e8eaf6', fontsize=8)
        self.line_pnl, = self.ax1.plot([], [], color='#00e676', linewidth=2.5, label='PnL', marker='o', markersize=3)
        self.ax1.axhline(y=0, color='#7c4dff', linestyle='--', linewidth=1.5, alpha=0.5)
        self.ax1.legend(loc='upper left', fontsize=8, facecolor='#1a1f3a', edgecolor='#00d4ff', labelcolor='#e8eaf6')
        
        # Setup subplot 2 - Win Rate & Trades
        self.ax2.set_title('Win Rate & Trade Count', color='#00d4ff', fontsize=10, pad=5, weight='bold')
        self.ax2.set_ylabel('Win Rate (%)', color='#00e676', fontsize=8)
        self.line_winrate, = self.ax2.plot([], [], color='#00e676', linewidth=2.5, label='Win Rate', marker='s', markersize=3)
        self.ax2_twin = self.ax2.twinx()
        self.ax2_twin.set_ylabel('Trades', color='#00b0ff', fontsize=8)
        self.ax2_twin.tick_params(colors='#e8eaf6', labelsize=8)
        self.ax2_twin.spines['right'].set_color('#7c4dff')
        self.line_trades, = self.ax2_twin.plot([], [], color='#00b0ff', linewidth=2.5, linestyle='--', label='Trades', marker='^', markersize=3)
        
        # Combine legends
        lines = [self.line_winrate, self.line_trades]
        labels = [l.get_label() for l in lines]
        self.ax2.legend(lines, labels, loc='upper left', fontsize=8, facecolor='#1a1f3a', edgecolor='#00d4ff', labelcolor='#e8eaf6')
        
        # Setup subplot 3 - Latency
        self.ax3.set_title('Tick Latency (μs)', color='#00d4ff', fontsize=10, pad=5, weight='bold')
        self.ax3.set_xlabel('Time', color='#e8eaf6', fontsize=8)
        self.ax3.set_ylabel('Latency', color='#e8eaf6', fontsize=8)
        self.line_latency, = self.ax3.plot([], [], color='#ff1744', linewidth=2.5, label='Latency', marker='D', markersize=3)
        self.ax3.legend(loc='upper left', fontsize=8, facecolor='#1a1f3a', edgecolor='#00d4ff', labelcolor='#e8eaf6')
        
        # Embed matplotlib figure in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Chart control buttons
        btn_frame = ttk.Frame(chart_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(btn_frame, text="🔄 Reset Chart", command=self.reset_chart).pack(side=tk.LEFT, padx=5)
        ttk.Label(btn_frame, text="📈 Real-time updates every 1 second", style='TLabel').pack(side=tk.RIGHT, padx=5)
    
    def build_risk_tab(self):
        """Build risk management tab"""
        # Clear any existing content
        for widget in self.risk_tab.winfo_children():
            widget.destroy()
        
        frame = ttk.Frame(self.risk_tab, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Risk limits
        limits_frame = ttk.LabelFrame(frame, text="Risk Limits Configuration", padding="10")
        limits_frame.pack(fill=tk.X, pady=(0, 10))
        
        limits = [
            ("Max Daily Loss ($):", "max_daily_loss", "1000"),
            ("Max Daily Trades:", "max_daily_trades", "500"),
            ("Max Position Size:", "max_position_size", "1.0"),
            ("Max Positions:", "max_positions", "3"),
            ("Max Drawdown (%):", "max_drawdown_pct", "10"),
        ]
        
        # Link existing limit_vars to entry widgets
        for idx, (label, key, default) in enumerate(limits):
            ttk.Label(limits_frame, text=label).grid(row=idx, column=0, sticky=tk.W, padx=5, pady=5)
            
            # Use the already initialized StringVar from __init__
            entry = ttk.Entry(limits_frame, textvariable=self.limit_vars[key], width=15)
            entry.grid(row=idx, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Update button
        ttk.Button(limits_frame, text="🔄 Update & Refresh Status", 
                  command=self.update_risk_status, width=25).grid(row=len(limits), column=0, columnspan=2, pady=13)
        
        # Risk Status Container
        status_container = ttk.Frame(frame)
        status_container.pack(fill=tk.BOTH, expand=True)
        
        # Left: Risk Metrics
        metrics_frame = ttk.LabelFrame(status_container, text="Risk Metrics & Status", padding="10")
        metrics_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.risk_metrics_text = scrolledtext.ScrolledText(metrics_frame, height=15, 
                                                           bg='#151932', fg='#e8eaf6', 
                                                           font=('Consolas', 10), insertbackground='#00d4ff')
        self.risk_metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # Right: Risk Events Log
        events_frame = ttk.LabelFrame(status_container, text="Risk Events & Alerts", padding="10")
        events_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.risk_status_text = scrolledtext.ScrolledText(events_frame, height=15, 
                                                          bg='#151932', fg='#e8eaf6', 
                                                          font=('Consolas', 9), insertbackground='#00d4ff')
        self.risk_status_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize risk status display
        # Clear any existing content
        for widget in self.ml_tab.winfo_children():
            widget.destroy()
        
        self.update_risk_status()
    
    def build_ml_tab(self):
        """Build ML models tab"""
        frame = ttk.Frame(self.ml_tab, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # ML configuration
        ml_config_frame = ttk.LabelFrame(frame, text="ML Configuration", padding="10")
        ml_config_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(ml_config_frame, text="Training Days:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        # Use the already initialized ml_days_var from __init__
        ttk.Entry(ml_config_frame, textvariable=self.ml_days_var, width=10).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5
        )
        
        ttk.Button(ml_config_frame, text="🎓 Train Models", 
                  command=self.train_ml_models, width=15).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(ml_config_frame, text="💾 Save Models", 
                  command=self.save_ml_models, width=15).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(ml_config_frame, text="📁 Load Models", 
                  command=self.load_ml_models, width=15).grid(row=0, column=4, padx=5, pady=5)
        
        ttk.Button(ml_config_frame, text="🔄 Refresh Status", 
                  command=self.update_ml_status, width=15).grid(row=0, column=5, padx=5, pady=5)
        
        # ML Status Display
        status_container = ttk.Frame(frame)
        status_container.pack(fill=tk.BOTH, expand=True)
        
        # Left: ML Status Info
        info_frame = ttk.LabelFrame(status_container, text="ML Status & Information", padding="10")
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.ml_info_text = scrolledtext.ScrolledText(info_frame, height=20, 
                                                      bg='#151932', fg='#e8eaf6', 
                                                      font=('Consolas', 10), insertbackground='#00d4ff')
        self.ml_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Right: ML Logs
        log_frame = ttk.LabelFrame(status_container, text="ML Training & Prediction Logs", padding="10")
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.ml_status_text = scrolledtext.ScrolledText(log_frame, height=20, 
                                                        bg='#151932', fg='#e8eaf6', 
                                                        font=('Consolas', 9), insertbackground='#00d4ff')
        self.ml_status_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize ML status display
        self.update_ml_status()
    
    def build_log_tab(self):
        """Build logs tab"""
        frame = ttk.Frame(self.log_tab, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Log viewer
        log_frame = ttk.LabelFrame(frame, text="System Logs", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, bg='#151932', fg='#e8eaf6', 
                                                  font=('Consolas', 9), insertbackground='#00d4ff')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(btn_frame, text="🗑️ Clear Logs", 
                  command=lambda: self.log_text.delete(1.0, tk.END), width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="💾 Export Logs", 
                  command=self.export_logs, width=15).pack(side=tk.LEFT, padx=5)
    
    def build_strategy_tab(self):
        """Build Strategy Tester tab"""
        # Clear existing content first
        for widget in self.strategy_tab.winfo_children():
            widget.destroy()
        
        frame = ttk.Frame(self.strategy_tab, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Test Parameters
        param_frame = ttk.LabelFrame(frame, text="🔧 Test Parameters", padding="15")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 0: Symbol and Date Range
        ttk.Label(param_frame, text="Symbol:", font=('Segoe UI', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_symbol_var = tk.StringVar(value=self.symbol_var.get())
        ttk.Entry(param_frame, textvariable=self.test_symbol_var, width=20, font=('Segoe UI', 10)).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(param_frame, text="Start Date:", font=('Segoe UI', 10, 'bold')).grid(row=0, column=2, sticky=tk.W, padx=(20, 5), pady=5)
        self.test_start_date = tk.StringVar(value=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
        ttk.Entry(param_frame, textvariable=self.test_start_date, width=15, font=('Segoe UI', 10)).grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(param_frame, text="End Date:", font=('Segoe UI', 10, 'bold')).grid(row=0, column=4, sticky=tk.W, padx=(20, 5), pady=5)
        self.test_end_date = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(param_frame, textvariable=self.test_end_date, width=15, font=('Segoe UI', 10)).grid(row=0, column=5, sticky=tk.W, padx=5)

        # Row 0.5: Initial Balance (Saldo Awal)
        ttk.Label(param_frame, text="Saldo Awal ($):", font=('Segoe UI', 10, 'bold')).grid(row=0, column=6, sticky=tk.W, padx=(20, 5), pady=5)
        self.test_initial_balance = tk.StringVar(value="10000")
        ttk.Entry(param_frame, textvariable=self.test_initial_balance, width=12, font=('Segoe UI', 10)).grid(row=0, column=7, sticky=tk.W, padx=5)
        

        # Row 1: Strategy Settings (expanded)
        ttk.Label(param_frame, text="Min Signal:", font=('Segoe UI', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_signal_var = tk.StringVar(value=self.signal_strength_var.get())
        ttk.Entry(param_frame, textvariable=self.test_signal_var, width=10, font=('Segoe UI', 10)).grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="SL Multiplier (ATR):", font=('Segoe UI', 10, 'bold')).grid(row=1, column=2, sticky=tk.W, padx=(20, 5), pady=5)
        self.test_sl_var = tk.StringVar(value=self.sl_multiplier_var.get())
        ttk.Entry(param_frame, textvariable=self.test_sl_var, width=10, font=('Segoe UI', 10)).grid(row=1, column=3, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Risk:Reward Ratio:", font=('Segoe UI', 10, 'bold')).grid(row=1, column=4, sticky=tk.W, padx=(20, 5), pady=5)
        self.test_rr_var = tk.StringVar(value=self.risk_reward_var.get())
        ttk.Entry(param_frame, textvariable=self.test_rr_var, width=10, font=('Segoe UI', 10)).grid(row=1, column=5, sticky=tk.W, padx=5)

        # Row 2: Risk & Money Management
        ttk.Label(param_frame, text="Max Volatility:", font=('Segoe UI', 10, 'bold')).grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_max_volatility = tk.StringVar(value=self.max_volatility_var.get())
        ttk.Entry(param_frame, textvariable=self.test_max_volatility, width=10, font=('Segoe UI', 10)).grid(row=2, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Filling Mode:", font=('Segoe UI', 10, 'bold')).grid(row=2, column=2, sticky=tk.W, padx=(20, 5), pady=5)
        self.test_filling_mode = tk.StringVar(value=self.filling_mode_var.get())
        ttk.Combobox(param_frame, textvariable=self.test_filling_mode, values=["FOK", "IOC", "RETURN"], width=10, state="readonly").grid(row=2, column=3, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Max Floating Loss ($):", font=('Segoe UI', 10, 'bold')).grid(row=2, column=4, sticky=tk.W, padx=(20, 5), pady=5)
        self.test_max_floating_loss = tk.StringVar(value=self.max_floating_loss_var.get())
        ttk.Entry(param_frame, textvariable=self.test_max_floating_loss, width=10, font=('Segoe UI', 10)).grid(row=2, column=5, sticky=tk.W, padx=5)

        # Row 3: Volume & Risk
        ttk.Label(param_frame, text="Default Volume:", font=('Segoe UI', 10, 'bold')).grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_default_volume = tk.StringVar(value=getattr(self, 'default_volume_var', tk.StringVar(value="0.01")).get())
        ttk.Entry(param_frame, textvariable=self.test_default_volume, width=10, font=('Segoe UI', 10)).grid(row=3, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Risk per Trade (%):", font=('Segoe UI', 10, 'bold')).grid(row=3, column=2, sticky=tk.W, padx=(20, 5), pady=5)
        self.test_risk_per_trade = tk.StringVar(value=getattr(self, 'risk_var', tk.StringVar(value="1.0")).get())
        ttk.Entry(param_frame, textvariable=self.test_risk_per_trade, width=10, font=('Segoe UI', 10)).grid(row=3, column=3, sticky=tk.W, padx=5)

        # Row 4: Daily & Position Limits
        ttk.Label(param_frame, text="Max Daily Loss ($):", font=('Segoe UI', 10, 'bold')).grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_max_daily_loss = tk.StringVar(value=getattr(self, 'limit_vars', {}).get('max_daily_loss', tk.StringVar(value="500")).get())
        ttk.Entry(param_frame, textvariable=self.test_max_daily_loss, width=10, font=('Segoe UI', 10)).grid(row=4, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Max Daily Trades:", font=('Segoe UI', 10, 'bold')).grid(row=4, column=2, sticky=tk.W, padx=(20, 5), pady=5)
        self.test_max_daily_trades = tk.StringVar(value=getattr(self, 'limit_vars', {}).get('max_daily_trades', tk.StringVar(value="260")).get())
        ttk.Entry(param_frame, textvariable=self.test_max_daily_trades, width=10, font=('Segoe UI', 10)).grid(row=4, column=3, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Max Position Size:", font=('Segoe UI', 10, 'bold')).grid(row=4, column=4, sticky=tk.W, padx=(20, 5), pady=5)
        self.test_max_position_size = tk.StringVar(value=getattr(self, 'limit_vars', {}).get('max_position_size', tk.StringVar(value="1.0")).get())
        ttk.Entry(param_frame, textvariable=self.test_max_position_size, width=10, font=('Segoe UI', 10)).grid(row=4, column=5, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Max Positions:", font=('Segoe UI', 10, 'bold')).grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_max_positions = tk.StringVar(value=getattr(self, 'limit_vars', {}).get('max_positions', tk.StringVar(value="50")).get())
        ttk.Entry(param_frame, textvariable=self.test_max_positions, width=10, font=('Segoe UI', 10)).grid(row=5, column=1, sticky=tk.W, padx=5)

        ttk.Label(param_frame, text="Max Drawdown (%):", font=('Segoe UI', 10, 'bold')).grid(row=5, column=2, sticky=tk.W, padx=(20, 5), pady=5)
        self.test_max_drawdown = tk.StringVar(value=getattr(self, 'limit_vars', {}).get('max_drawdown_pct', tk.StringVar(value="99")).get())
        ttk.Entry(param_frame, textvariable=self.test_max_drawdown, width=10, font=('Segoe UI', 10)).grid(row=5, column=3, sticky=tk.W, padx=5)

        # Row 6: Use ML and Run Button
        self.test_use_ml = tk.BooleanVar(value=self.use_ml_var.get())
        ttk.Checkbutton(param_frame, text="✓ Use ML Predictions", variable=self.test_use_ml, 
                   style='Switch.TCheckbutton').grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)

        ttk.Button(param_frame, text="🧪 Run Backtest", command=self.run_backtest, 
              width=25).grid(row=6, column=4, columnspan=2, padx=5, pady=10, sticky=tk.E)
        
        # Progress Bar Section
        progress_frame = ttk.LabelFrame(frame, text="📊 Backtest Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Progress bar
        self.backtest_progress = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.backtest_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Progress label
        self.backtest_progress_label = ttk.Label(progress_frame, text="Ready to start backtest", 
                                                font=('Segoe UI', 10))
        self.backtest_progress_label.pack(side=tk.LEFT)
        
        # Results Container with proper expansion
        results_container = ttk.Frame(frame)
        results_container.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Configure grid weights for proper resizing
        results_container.grid_columnconfigure(0, weight=1)
        results_container.grid_columnconfigure(1, weight=1)
        results_container.grid_rowconfigure(0, weight=1)
        
        # Left: Statistics
        stats_frame = ttk.LabelFrame(results_container, text="📊 Backtest Results", padding="10")
        stats_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(0, 5))
        
        self.backtest_stats_text = scrolledtext.ScrolledText(stats_frame, height=20, width=50,
                                                             bg='#151932', fg='#e8eaf6',
                                                             font=('Consolas', 10), insertbackground='#00d4ff',
                                                             wrap=tk.WORD)
        self.backtest_stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Right: Trade List
        trades_frame = ttk.LabelFrame(results_container, text="📝 Trade History", padding="10")
        trades_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(5, 0))
        
        self.backtest_trades_text = scrolledtext.ScrolledText(trades_frame, height=20, width=50,
                                                              bg='#151932', fg='#e8eaf6',
                                                              font=('Consolas', 9), insertbackground='#00d4ff',
                                                              wrap=tk.NONE)
        self.backtest_trades_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.backtest_stats_text.insert(1.0, "📋 No backtest results yet.\n\n✨ Configure your test parameters above and click 'Run Backtest' to start testing your strategy.\n\n💡 Tips:\n- Test period should be at least 7 days\n- Use realistic settings from your live trading\n- Enable ML if you have trained models")
        
        self.log_message("Strategy Tester tab initialized", "INFO")
    
    def run_backtest(self):
        """Run strategy backtest with current settings"""
        try:
            # Clear previous results
            self.backtest_stats_text.delete(1.0, tk.END)
            self.backtest_trades_text.delete(1.0, tk.END)
            
            # Reset progress
            self.backtest_progress['value'] = 0
            self.backtest_progress_label.config(text="⏳ Initializing backtest...")
            
            self.backtest_stats_text.insert(tk.END, "🗿 STRATEGY BACKTEST\n")
            self.backtest_stats_text.insert(tk.END, "=" * 60 + "\n\n")
            self.backtest_stats_text.insert(tk.END, "⏳ Preparing backtest environment...\n")
            self.root.update_idletasks()
            

            # Get ALL parameters from Strategy Tester UI
            symbol = self.test_symbol_var.get()
            start_date = datetime.strptime(self.test_start_date.get(), "%Y-%m-%d")
            end_date = datetime.strptime(self.test_end_date.get(), "%Y-%m-%d")
            min_signal = float(self.test_signal_var.get())
            sl_multi = float(self.test_sl_var.get())
            rr_ratio = float(self.test_rr_var.get())
            use_ml = self.test_use_ml.get()
            # Risk & Money Management
            max_volatility = float(self.test_max_volatility.get())
            filling_mode = self.test_filling_mode.get()
            max_floating_loss = float(self.test_max_floating_loss.get())
            default_volume = float(self.test_default_volume.get())
            risk_per_trade = float(self.test_risk_per_trade.get())
            max_daily_loss = float(self.test_max_daily_loss.get())
            max_daily_trades = int(self.test_max_daily_trades.get())
            max_position_size = float(self.test_max_position_size.get())
            max_positions = int(self.test_max_positions.get())
            max_drawdown = float(self.test_max_drawdown.get())
            initial_balance = float(self.test_initial_balance.get())
            # (Catatan: variabel di atas sudah siap dipakai untuk logika backtest berikutnya)
            
            # Progress update
            self.backtest_progress['value'] = 5
            self.backtest_progress_label.config(text="🔧 Loading core modules...")
            self.root.update_idletasks()
            
            # Load core modules if needed
            if not self.core_modules_loaded:
                self.load_core_modules()
            
            # Progress update
            self.backtest_progress['value'] = 10
            self.backtest_progress_label.config(text="📊 Fetching historical data...")
            self.backtest_stats_text.insert(tk.END, "\n📊 Fetching historical data from MT5...\n")
            self.root.update_idletasks()
            
            # Run backtest in thread to avoid blocking UI
            import threading
            def backtest_thread():
                try:
                    # Import BacktestEngine
                    try:
                        from backtest_engine import BacktestEngine
                    except ImportError:
                        self.root.after(0, lambda: self.update_backtest_status("❌ backtest_engine.py not found", error=True))
                        return
                    
                    # Initialize MT5
                    import MetaTrader5 as mt5
                    
                    if not mt5.initialize():
                        self.root.after(0, lambda: self.update_backtest_status("❌ Failed to initialize MT5", error=True))
                        return
                    
                    # Progress update
                    self.root.after(0, lambda: self.update_backtest_progress(15, "🔍 Creating backtest engine..."))
                    
                    # Create BacktestEngine instance
                    engine = BacktestEngine(symbol=symbol, initial_balance=initial_balance)
                    
                    # Progress update
                    self.root.after(0, lambda: self.update_backtest_progress(25, "📊 Loading OHLCV data..."))
                    self.root.after(0, lambda: self.backtest_stats_text.insert(tk.END, "📊 Loading OHLCV bars from MT5...\n"))
                    
                    # Load data using BacktestEngine
                    df = engine.load_data(symbol, start_date, end_date, mt5.TIMEFRAME_M5)
                    
                    if df is None or len(df) == 0:
                        self.root.after(0, lambda: self.update_backtest_status("❌ No data available for this period", error=True))
                        mt5.shutdown()
                        return
                    
                    # Progress update
                    self.root.after(0, lambda: self.update_backtest_progress(35, "✅ Loaded {} bars".format(len(df))))
                    self.root.after(0, lambda: self.backtest_stats_text.insert(tk.END, "✅ Retrieved {} bars of data\n".format(len(df))))
                    
                    # Define strategy function for backtest
                    def strategy_func(df_segment):
                        """
                        Strategy function that receives dataframe segment up to current bar.
                        Returns 1 for BUY, -1 for SELL, 0 for NO SIGNAL
                        """
                        if len(df_segment) < 20:
                            return 0
                        
                        # Simple SMA crossover strategy with min signal strength filter
                        sma_5 = df_segment['close'].iloc[-5:].mean()
                        sma_20 = df_segment['close'].iloc[-20:].mean()
                        current_price = df_segment['close'].iloc[-1]
                        
                        # BUY: Fast SMA > Slow SMA with minimum signal strength
                        if sma_5 > sma_20:
                            signal_strength = (sma_5 - sma_20) / current_price * 1000
                            if signal_strength > min_signal:
                                return 1
                        
                        # SELL: Fast SMA < Slow SMA with minimum signal strength  
                        elif sma_5 < sma_20:
                            signal_strength = (sma_20 - sma_5) / current_price * 1000
                            if signal_strength > min_signal * 0.8:  # Slightly lower threshold for shorts
                                return -1
                        
                        return 0
                    
                    # Progress update
                    self.root.after(0, lambda: self.update_backtest_progress(45, "⚙️ Running backtest simulation..."))
                    self.root.after(0, lambda: self.backtest_stats_text.insert(tk.END, "\n⚙️ Simulating trades with realistic position management...\n"))
                    
                    # Run backtest with engine
                    metrics = engine.run_backtest(
                        df, 
                        strategy_func, 
                        commission=0.0002,  # 0.02% per side
                        slippage=0.00001     # 0.1 pips
                    )
                    
                    mt5.shutdown()
                    
                    # Progress update
                    self.root.after(0, lambda: self.update_backtest_progress(90, "📈 Calculating performance metrics..."))
                    
                    # Display results using BacktestMetrics
                    def show_results():
                        self.backtest_stats_text.delete(1.0, tk.END)
                        self.backtest_stats_text.insert(tk.END, "🗿 BACKTEST RESULTS\n")
                        self.backtest_stats_text.insert(tk.END, "=" * 60 + "\n\n")
                        self.backtest_stats_text.insert(tk.END, f"Symbol: {symbol}\n")
                        self.backtest_stats_text.insert(tk.END, f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
                        self.backtest_stats_text.insert(tk.END, f"Bars analyzed: {len(df)}\n")
                        self.backtest_stats_text.insert(tk.END, f"Timeframe: M5\n\n")
                        
                        self.backtest_stats_text.insert(tk.END, "📊 PERFORMANCE METRICS:\n")
                        self.backtest_stats_text.insert(tk.END, "-" * 60 + "\n")
                        self.backtest_stats_text.insert(tk.END, f"Total Trades: {metrics.total_trades}\n")
                        self.backtest_stats_text.insert(tk.END, f"Winning: {metrics.winning_trades} | Losing: {metrics.losing_trades}\n")
                        self.backtest_stats_text.insert(tk.END, f"Win Rate: {metrics.win_rate:.2f}%\n")
                        self.backtest_stats_text.insert(tk.END, f"Gross Profit: ${metrics.gross_profit:.2f}\n")
                        self.backtest_stats_text.insert(tk.END, f"Gross Loss: ${metrics.gross_loss:.2f}\n")
                        self.backtest_stats_text.insert(tk.END, f"Net Profit/Loss: ${metrics.net_profit:.2f}\n")
                        self.backtest_stats_text.insert(tk.END, f"ROI: {metrics.total_return_pct:.2f}%\n")
                        self.backtest_stats_text.insert(tk.END, f"Initial Balance: ${initial_balance:.2f}\n")
                        self.backtest_stats_text.insert(tk.END, f"Final Balance: ${metrics.final_balance:.2f}\n\n")
                        
                        self.backtest_stats_text.insert(tk.END, "💰 TRADE ANALYSIS:\n")
                        self.backtest_stats_text.insert(tk.END, "-" * 60 + "\n")
                        self.backtest_stats_text.insert(tk.END, f"Avg Win: ${metrics.avg_win:.2f}\n")
                        self.backtest_stats_text.insert(tk.END, f"Avg Loss: ${metrics.avg_loss:.2f}\n")
                        self.backtest_stats_text.insert(tk.END, f"Profit Factor: {metrics.profit_factor:.2f}\n")
                        self.backtest_stats_text.insert(tk.END, f"Best Trade: ${metrics.best_trade:.2f}\n")
                        self.backtest_stats_text.insert(tk.END, f"Worst Trade: ${metrics.worst_trade:.2f}\n")
                        self.backtest_stats_text.insert(tk.END, f"Avg Trade Duration: {metrics.avg_trade_duration:.1f} minutes\n\n")
                        
                        self.backtest_stats_text.insert(tk.END, "📈 RISK METRICS:\n")
                        self.backtest_stats_text.insert(tk.END, "-" * 60 + "\n")
                        self.backtest_stats_text.insert(tk.END, f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%\n")
                        self.backtest_stats_text.insert(tk.END, f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n")
                        self.backtest_stats_text.insert(tk.END, f"Sortino Ratio: {metrics.sortino_ratio:.2f}\n")
                        self.backtest_stats_text.insert(tk.END, f"Calmar Ratio: {metrics.calmar_ratio:.2f}\n\n")
                        
                        self.backtest_stats_text.insert(tk.END, "🎯 STRATEGY SETTINGS:\n")
                        self.backtest_stats_text.insert(tk.END, "-" * 60 + "\n")
                        self.backtest_stats_text.insert(tk.END, f"Min Signal Strength: {min_signal}\n")
                        self.backtest_stats_text.insert(tk.END, f"SL Multiplier: {sl_multi}\n")
                        self.backtest_stats_text.insert(tk.END, f"Risk:Reward Ratio: {rr_ratio}\n")
                        self.backtest_stats_text.insert(tk.END, f"Commission (per side): 0.02%\n")
                        self.backtest_stats_text.insert(tk.END, f"Slippage: 0.1 pips\n\n")
                        
                        # Show trades
                        self.backtest_trades_text.delete(1.0, tk.END)
                        self.backtest_trades_text.insert(tk.END, "📋 TRADE HISTORY\n")
                        self.backtest_trades_text.insert(tk.END, "=" * 120 + "\n\n")
                        
                        if metrics.total_trades == 0:
                            self.backtest_trades_text.insert(tk.END, "⚠️ No trades generated during this backtest.\n\n")
                            self.backtest_trades_text.insert(tk.END, "💡 Suggestions to generate more signals:\n")
                            self.backtest_trades_text.insert(tk.END, "  • Lower min signal strength (try 0.3-0.4)\n")
                            self.backtest_trades_text.insert(tk.END, "  • Increase test period (try 30-90 days)\n")
                            self.backtest_trades_text.insert(tk.END, "  • Check if symbol has sufficient volatility\n")
                            self.backtest_trades_text.insert(tk.END, "  • Verify MT5 connection and data availability\n")
                        else:
                            header = f"{'#':<4} {'Entry Time':<20} {'Exit Time':<20} {'Type':<6} {'Entry Price':<12} {'Exit Price':<12} {'P&L':<12} {'Status':<8}\n"
                            self.backtest_trades_text.insert(tk.END, header)
                            self.backtest_trades_text.insert(tk.END, "-" * 120 + "\n")
                            
                            # Display first 100 trades
                            for idx, trade in enumerate(engine.trades[:100], 1):
                                entry_time = trade.entry_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(trade.entry_time, 'strftime') else str(trade.entry_time)
                                exit_time = trade.exit_time.strftime('%Y-%m-%d %H:%M:%S') if trade.exit_time and hasattr(trade.exit_time, 'strftime') else (str(trade.exit_time) if trade.exit_time else "OPEN")
                                status = "✓ WIN" if trade.win else "✗ LOSS"
                                
                                line = f"{idx:<4} {entry_time:<20} {exit_time:<20} {trade.trade_type:<6} {trade.entry_price:<12.5f} {(trade.exit_price if trade.exit_price else 0):<12.5f} ${trade.pnl_after_commission:<11.2f} {status:<8}\n"
                                self.backtest_trades_text.insert(tk.END, line)
                            
                            if len(engine.trades) > 100:
                                self.backtest_trades_text.insert(tk.END, f"\n... and {len(engine.trades)-100} more trades (export CSV for full history)\n")
                        
                        self.backtest_stats_text.insert(tk.END, "\n✅ Backtest completed successfully!\n")
                    
                    self.root.after(0, show_results)
                    
                    # Export results to CSV
                    try:
                        engine.export_results("backtest_results.csv")
                        self.root.after(0, lambda: self.backtest_stats_text.insert(tk.END, "📁 Results exported to backtest_results.csv\n"))
                    except Exception as e:
                        self.root.after(0, lambda: self.backtest_stats_text.insert(tk.END, f"⚠️ Could not export CSV: {str(e)}\n"))
                    
                except Exception as e:
                    self.root.after(0, lambda: self.update_backtest_status(f"❌ Error: {str(e)}", error=True))
            
            thread = threading.Thread(target=backtest_thread, daemon=True)
            thread.start()
            
        except Exception as e:
            self.backtest_stats_text.insert(tk.END, f"\n❌ Error: {str(e)}\n")
            self.backtest_progress_label.config(text="❌ Backtest failed")
    
    def update_backtest_progress(self, value, text):
        """Update backtest progress bar and label"""
        self.backtest_progress['value'] = value
        self.backtest_progress_label.config(text=text)
        self.root.update_idletasks()
    
    def update_backtest_status(self, message, error=False):
        """Update backtest status with message"""
        self.backtest_stats_text.insert(tk.END, f"\n{message}\n")
        if error:
            self.backtest_progress['value'] = 0
            self.backtest_progress_label.config(text="❌ Backtest failed")
        self.root.update_idletasks()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def log_message(self, message: str, level: str = "INFO"):
        """Add message to logs"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        # Add to status text if it exists
        if hasattr(self, 'status_text'):
            self.status_text.insert(tk.END, log_entry)
            self.status_text.see(tk.END)
        
        # Add to log text if it exists
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
            
            # Color coding
            if level == "ERROR":
                self.log_text.tag_add("error", f"{self.log_text.index(tk.END)}-2l", f"{self.log_text.index(tk.END)}-1l")
                self.log_text.tag_config("error", foreground="#ff1744", font=('Consolas', 9, 'bold'))
            elif level == "SUCCESS":
                self.log_text.tag_add("success", f"{self.log_text.index(tk.END)}-2l", f"{self.log_text.index(tk.END)}-1l")
                self.log_text.tag_config("success", foreground="#00e676", font=('Consolas', 9, 'bold'))
    
    def start_trading(self):
        """Start the HFT trading engine"""
        try:
            # Load core modules if not already loaded
            if not self.core_modules_loaded:
                self.load_core_modules()
            
            self.log_message("Initializing trading engine...", "INFO")
            
            # Get configuration
            config = {
                'magic_number': int(self.magic_var.get()),
                'default_volume': float(self.volume_var.get()),
                'risk_per_trade': float(self.risk_var.get()) / 100,
                'min_signal_strength': float(self.signal_strength_var.get()),
                'max_spread': float(self.max_spread_var.get()),
                'max_volatility': float(self.max_volatility_var.get()),
                'filling_mode': self.filling_mode_var.get(),
                'sl_multiplier': float(self.sl_multiplier_var.get()),
                'risk_reward_ratio': float(self.risk_reward_var.get()),
                'tp_mode': self.tp_mode_var.get(),
                'tp_dollar_amount': float(self.tp_dollar_var.get()),
                'max_floating_loss': float(self.max_floating_loss_var.get()),
                'max_floating_profit': float(self.max_floating_profit_var.get()),
                'mt5_path': self.mt5_path_var.get(),
                
                # Add missing thresholds with defaults based on symbol
                'min_delta_threshold': 20 if 'GOLD' in self.symbol_var.get().upper() or 'XAU' in self.symbol_var.get().upper() else 50,
                'min_velocity_threshold': 0.0001 if 'GOLD' in self.symbol_var.get().upper() or 'XAU' in self.symbol_var.get().upper() else 0.00001,
                'analysis_interval': 0.1,
                'slippage': 50 if 'GOLD' in self.symbol_var.get().upper() or 'XAU' in self.symbol_var.get().upper() else 20,
                
                'max_daily_loss': float(self.limit_vars['max_daily_loss'].get()),
                'max_daily_trades': int(self.limit_vars['max_daily_trades'].get()),
                'max_position_size': float(self.limit_vars['max_position_size'].get()),
                'max_positions': int(self.limit_vars['max_positions'].get()),
                'max_drawdown_pct': float(self.limit_vars['max_drawdown_pct'].get()),
            }
            
            # Initialize components
            symbol = self.symbol_var.get()
            
            self.risk_manager = RiskManager(config)
            self.engine = UltraLowLatencyEngine(symbol, config, self.risk_manager)
            
            # Initialize ML if enabled
            if self.use_ml_var.get():
                self.ml_predictor = MLPredictor(symbol, config)
                try:
                    self.ml_predictor.load_models("./models")
                    self.log_message("ML models loaded successfully", "SUCCESS")
                except:
                    self.log_message("ML models not found. Train models first.", "ERROR")
                    self.use_ml_var.set(False)
            
            # Start engine
            if self.engine.start():
                self.is_running = True
                self.start_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)
                self.status_bar.config(text="Status: TRADING ACTIVE")
                
                self.log_message("✓ Trading engine started successfully!", "SUCCESS")
                self.log_message(f"Symbol: {symbol} | Volume: {config['default_volume']}", "INFO")
                self.log_message(f"Max Spread: {config['max_spread']} | Max Volatility: {config['max_volatility']}", "INFO")
                self.log_message(f"Min Signal Strength: {config['min_signal_strength']}", "INFO")
                self.log_message(f"Delta Threshold: {config['min_delta_threshold']} | Velocity: {config['min_velocity_threshold']}", "INFO")
                self.log_message(f"Filling Mode: {config['filling_mode']}", "INFO")
                
                # Log to risk management
                self.risk_log_message("✅ Trading session started", "SUCCESS")
                self.risk_log_message(f"Symbol: {symbol} | Volume: {config['default_volume']}", "INFO")
                self.risk_log_message(f"Max Daily Loss: ${config['max_daily_loss']:.2f}", "INFO")
                self.risk_log_message(f"Max Daily Trades: {config['max_daily_trades']}", "INFO")
                self.risk_log_message(f"Max Drawdown: {config['max_drawdown_pct']:.2f}%", "INFO")
                self.risk_log_message("Monitoring risk limits...", "INFO")
                
                # Start update thread
                self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
                self.update_thread.start()
            else:
                self.log_message("Failed to start trading engine", "ERROR")
                
        except Exception as e:
            self.log_message(f"Error starting engine: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Failed to start trading:\n{str(e)}")
    
    def stop_trading(self):
        """Stop the trading engine"""
        try:
            self.log_message("Stopping trading engine...", "INFO")
            
            self.is_running = False
            
            if self.engine:
                self.engine.stop()
            
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_bar.config(text="Status: Stopped")
            
            self.log_message("✓ Trading engine stopped", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"Error stopping engine: {str(e)}", "ERROR")
    
    def update_loop(self):
        """Update GUI with real-time data"""
        update_counter = 0
        while self.is_running:
            try:
                if self.engine and hasattr(self, 'metric_labels'):
                    # Get performance stats
                    stats = self.engine.get_performance_stats()
                    # Update metrics
                    self.metric_labels['tick_latency'].config(
                        text=f"{stats['tick_latency_avg_us']:.1f} μs"
                    )
                    self.metric_labels['exec_time'].config(
                        text=f"{stats['execution_time_avg_ms']:.2f} ms"
                    )
                    self.metric_labels['ticks_processed'].config(
                        text=f"{stats['ticks_processed']:,}"
                    )
                    self.metric_labels['current_position'].config(
                        text=stats['current_position'] or "None"
                    )
                    # Get risk stats
                    if self.risk_manager:
                        risk_summary = self.risk_manager.get_trading_summary()
                        self.metric_labels['trades_today'].config(
                            text=f"{risk_summary['daily_trades']}"
                        )
                        self.metric_labels['daily_pnl'].config(
                            text=f"${risk_summary['daily_pnl']:.2f}"
                        )
                        self.metric_labels['win_rate'].config(
                            text=f"{risk_summary['win_rate']*100:.1f}%"
                        )
                        # Update chart data
                        current_time = datetime.now()
                        self.chart_data['timestamps'].append(current_time)
                        self.chart_data['pnl'].append(risk_summary['daily_pnl'])
                        self.chart_data['equity'].append(risk_summary.get('equity', 0))
                        self.chart_data['trades'].append(risk_summary['daily_trades'])
                        self.chart_data['win_rate'].append(risk_summary['win_rate'] * 100)
                        self.chart_data['latency'].append(stats['tick_latency_avg_us'])
                        # Update chart
                        self.update_chart()
                        # Update risk status every 5 seconds
                        if update_counter % 5 == 0:
                            self.root.after(0, self.update_risk_status)
                update_counter += 1
                threading.Event().wait(1.0)  # Update every second
                
            except Exception as e:
                print(f"Update error: {e}")
    
    def train_ml_models(self):
        """Train ML models"""
        # Validate training days
        try:
            days = int(self.ml_days_var.get())
            if days > 90:
                messagebox.showerror(
                    self.t('training_days_error'),
                    self.t('training_days_max')
                )
                return
            if days < 1:
                messagebox.showerror(
                    self.t('training_days_error'),
                    "Training days must be at least 1 day." if self.current_language.get() == 'EN' 
                    else "Training days harus minimal 1 hari."
                )
                return
        except ValueError:
            messagebox.showerror(
                self.t('training_days_error'),
                "Please enter a valid number for training days." if self.current_language.get() == 'EN'
                else "Mohon masukkan angka yang valid untuk training days."
            )
            return
        
        # Load core modules if not already loaded
        if not self.core_modules_loaded:
            self.load_core_modules()
        
        # Ensure ML tab is built so logs can be displayed
        if not hasattr(self, 'ml_info_text'):
            self.log_message("Building ML interface...", "INFO")
            self.build_ml_tab()
            # Switch to ML tab to show progress
            self.notebook.select(self.ml_tab)
        
        self.log_message("Starting ML training...", "INFO")
        self.ml_log_message("🎓 Starting ML model training...", "INFO")
        
        def train_thread():
            try:
                self.ml_log_message(f"Initializing MT5 connection...", "INFO")
                if not mt5.initialize():
                    self.log_message("Failed to initialize MT5", "ERROR")
                    self.ml_log_message("✗ Failed to initialize MT5", "ERROR")
                    return
                
                symbol = self.symbol_var.get()
                days = int(self.ml_days_var.get())
                
                self.ml_log_message(f"Symbol: {symbol} | Training days: {days}", "INFO")
                self.ml_log_message("Downloading historical data...", "INFO")
                
                predictor = MLPredictor(symbol, {})
                
                self.ml_log_message("Training LSTM model...", "INFO")
                self.ml_log_message("Training GRU model...", "INFO")
                self.ml_log_message("Training CNN model...", "INFO")
                
                if predictor.train(days):
                    self.ml_log_message("✓ All models trained successfully!", "SUCCESS")
                    
                    # Get training stats
                    stats = predictor.get_training_stats()
                    
                    # Calculate recommended settings based on performance
                    test_acc = stats.get('test_accuracy', 0.6)
                    overfitting_gap = stats.get('train_accuracy', 0.7) - test_acc
                    
                    # Recommendations based on accuracy
                    if test_acc >= 0.62:
                        rec_signal = 0.5
                        rec_sl = 2.0
                    elif test_acc >= 0.59:
                        rec_signal = 0.55
                        rec_sl = 2.2
                    else:
                        rec_signal = 0.6
                        rec_sl = 2.5
                    
                    # Adjust for overfitting
                    if overfitting_gap > 12:
                        rec_signal += 0.05  # More conservative if overfitting
                    
                    # Save metadata
                    metadata = {
                        'symbol': symbol,
                        'training_days': days,
                        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'test_accuracy': test_acc,
                        'train_accuracy': stats.get('train_accuracy', 0.7),
                        'confidence_accuracy': stats.get('confidence_accuracy', 0.6),
                        'overfitting_gap': overfitting_gap,
                        'samples': stats.get('samples', 0),
                        'features': stats.get('features', 39),
                        'top_features': stats.get('top_features', []),
                        'recommended_settings': {
                            'min_signal_strength': rec_signal,
                            'sl_multiplier': rec_sl,
                            'risk_reward': 2.0,
                            'max_spread': 0.03 if 'GOLD' in symbol else 0.0005,
                            'enable_ml': True
                        }
                    }
                    
                    # Save to symbol-specific folder
                    symbol_folder = f"./models/{symbol}"
                    import os
                    import json
                    os.makedirs(symbol_folder, exist_ok=True)
                    
                    # Save metadata
                    with open(f"{symbol_folder}/metadata.json", 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    self.ml_log_message(f"Saving models to {symbol_folder}/...", "INFO")
                    predictor.save_models(symbol_folder)
                    self.log_message(f"✓ ML models trained and saved to {symbol_folder}!", "SUCCESS")
                    self.ml_log_message(f"✓ Models saved to {symbol_folder}", "SUCCESS")
                    self.ml_log_message(f"✓ Metadata saved (Test Acc: {test_acc:.2%}, Rec. Signal: {rec_signal})", "INFO")
                    
                    self.ml_predictor = predictor
                    
                    # Update status display
                    self.root.after(0, self.update_ml_status)
                else:
                    self.log_message("ML training failed", "ERROR")
                    self.ml_log_message("✗ Training failed - check data availability", "ERROR")
                
                mt5.shutdown()
                self.ml_log_message("Training session completed", "INFO")
                
            except Exception as e:
                self.log_message(f"ML training error: {str(e)}", "ERROR")
                self.ml_log_message(f"✗ Error during training: {str(e)}", "ERROR")
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def save_ml_models(self):
        """Save ML models to symbol-specific folder"""
        if self.ml_predictor:
            symbol = self.symbol_var.get()
            symbol_folder = f"./models/{symbol}"
            
            import os
            os.makedirs(symbol_folder, exist_ok=True)
            
            self.ml_log_message(f"Saving ML models to {symbol_folder}...", "INFO")
            if self.ml_predictor.save_models(symbol_folder):
                self.log_message(f"✓ ML models saved to {symbol_folder}", "SUCCESS")
                self.ml_log_message(f"✓ Models saved to {symbol_folder}", "SUCCESS")
                self.update_ml_status()
            else:
                self.log_message("Failed to save models", "ERROR")
                self.ml_log_message("✗ Failed to save models", "ERROR")
        else:
            self.log_message("No models to save", "ERROR")
            self.ml_log_message("✗ No models loaded to save", "ERROR")
    
    def load_ml_models(self):
        """Load ML models with selection dialog"""
        try:
            # Load core modules if not already loaded
            if not self.core_modules_loaded:
                self.load_core_modules()
            
            import os
            
            # Get available model folders
            models_dir = "./models"
            if not os.path.exists(models_dir):
                self.log_message("No models directory found", "ERROR")
                self.ml_log_message("✗ No models directory found", "ERROR")
                messagebox.showerror("Error", "No models directory found. Train models first.")
                return
            
            # Find all folders containing model files
            available_symbols = []
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path):
                    # Check if folder contains model files
                    has_models = any(
                        os.path.exists(os.path.join(item_path, f)) 
                        for f in ['lstm_model.h5', 'gru_model.h5', 'cnn_model.h5', 
                                 'direction_model.pkl', 'confidence_model.pkl']
                    )
                    if has_models:
                        available_symbols.append(item)
            
            if not available_symbols:
                self.log_message("No trained models found", "ERROR")
                self.ml_log_message("✗ No trained models found", "ERROR")
                messagebox.showinfo("No Models", "No trained models found.\n\nTrain models first by clicking '🎓 Train Models'.")
                return
            
            # Show selection dialog
            selection_dialog = tk.Toplevel(self.root)
            selection_dialog.title("Select Model to Load")
            selection_dialog.geometry("400x300")
            selection_dialog.configure(bg='#0a0e27')
            selection_dialog.transient(self.root)
            selection_dialog.grab_set()
            
            # Center dialog
            selection_dialog.update_idletasks()
            x = (selection_dialog.winfo_screenwidth() // 2) - (400 // 2)
            y = (selection_dialog.winfo_screenheight() // 2) - (300 // 2)
            selection_dialog.geometry(f"400x300+{x}+{y}")
            
            # Title
            title_label = ttk.Label(selection_dialog, text="📁 Select Model to Load", 
                                   style='Header.TLabel')
            title_label.pack(pady=10)
            
            # Instructions
            info_label = ttk.Label(selection_dialog, 
                                  text="Select which symbol's trained model to load:\n(Double-click to load)",
                                  style='TLabel')
            info_label.pack(pady=5)
            
            # Listbox with scrollbar
            list_frame = ttk.Frame(selection_dialog)
            list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                               bg='#151932', fg='#e8eaf6', font=('Consolas', 11),
                               selectmode=tk.SINGLE, activestyle='dotbox',
                               selectbackground='#7c4dff', selectforeground='#ffffff')
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=listbox.yview)
            
            # Add symbols to listbox
            for symbol in sorted(available_symbols):
                listbox.insert(tk.END, f"  {symbol}")
            
            # Select current symbol if available
            current_symbol = self.symbol_var.get()
            if current_symbol in available_symbols:
                idx = sorted(available_symbols).index(current_symbol)
                listbox.selection_set(idx)
                listbox.see(idx)
            else:
                listbox.selection_set(0)
            
            selected_symbol = [None]
            
            def on_load():
                selection = listbox.curselection()
                if selection:
                    selected_symbol[0] = sorted(available_symbols)[selection[0]]
                    selection_dialog.destroy()
                else:
                    messagebox.showwarning("No Selection", "Please select a model to load.")
            
            def on_double_click(event):
                """Handle double-click on listbox item"""
                selection = listbox.curselection()
                if selection:
                    selected_symbol[0] = sorted(available_symbols)[selection[0]]
                    selection_dialog.destroy()
            
            def on_cancel():
                selection_dialog.destroy()
            
            # Bind double-click event
            listbox.bind('<Double-Button-1>', on_double_click)
            
            # Buttons
            btn_frame = ttk.Frame(selection_dialog)
            btn_frame.pack(pady=10)
            
            ttk.Button(btn_frame, text="📥 Load", command=on_load, width=15).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="❌ Cancel", command=on_cancel, width=15).pack(side=tk.LEFT, padx=5)
            
            # Wait for dialog to close
            self.root.wait_window(selection_dialog)
            
            # Load selected model
            if selected_symbol[0]:
                symbol_folder = f"./models/{selected_symbol[0]}"
                self.ml_log_message(f"Loading models from {symbol_folder}...", "INFO")
                
                self.ml_predictor = MLPredictor(selected_symbol[0], {})
                
                if self.ml_predictor.load_models(symbol_folder):
                    self.log_message(f"✓ ML models loaded from {symbol_folder}", "SUCCESS")
                    self.ml_log_message(f"✓ Models loaded: {selected_symbol[0]}", "SUCCESS")
                    
                    # Update symbol in GUI
                    self.symbol_var.set(selected_symbol[0])
                    
                    # Load and apply metadata/recommended settings
                    import json
                    metadata_path = f"{symbol_folder}/metadata.json"
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            # Show metadata info
                            test_acc = metadata.get('test_accuracy', 0) * 100
                            train_acc = metadata.get('train_accuracy', 0) * 100
                            conf_acc = metadata.get('confidence_accuracy', 0) * 100
                            training_date = metadata.get('training_date', 'Unknown')
                            training_days = metadata.get('training_days', 40)
                            
                            self.ml_log_message(f"📊 Model Info:", "INFO")
                            self.ml_log_message(f"  • Trained: {training_date}", "INFO")
                            self.ml_log_message(f"  • Training days: {training_days}", "INFO")
                            self.ml_log_message(f"  • Test accuracy: {test_acc:.2f}%", "INFO")
                            self.ml_log_message(f"  • Train accuracy: {train_acc:.2f}%", "INFO")
                            self.ml_log_message(f"  • Confidence: {conf_acc:.2f}%", "INFO")
                            
                            # Ask user if they want to apply recommended settings
                            rec_settings = metadata.get('recommended_settings', {})
                            if rec_settings:
                                rec_signal = rec_settings.get('min_signal_strength', 0.5)
                                rec_sl = rec_settings.get('sl_multiplier', 2.0)
                                rec_rr = rec_settings.get('risk_reward', 2.0)
                                rec_spread = rec_settings.get('max_spread', 0.03)
                                
                                msg = f"Model loaded successfully!\\n\\n"
                                msg += f"Symbol: {selected_symbol[0]}\\n"
                                msg += f"Test Accuracy: {test_acc:.2f}%\\n"
                                msg += f"Trained: {training_date}\\n\\n"
                                msg += f"📋 Recommended Settings:\\n"
                                msg += f"  • Min Signal Strength: {rec_signal}\\n"
                                msg += f"  • SL Multiplier: {rec_sl}\\n"
                                msg += f"  • Risk:Reward: {rec_rr}\\n"
                                msg += f"  • Max Spread: {rec_spread}\\n\\n"
                                msg += f"Apply these settings?"
                                
                                if messagebox.askyesno("Apply Recommended Settings?", msg):
                                    # Apply recommended settings
                                    self.signal_strength_var.set(str(rec_signal))
                                    self.sl_multiplier_var.set(str(rec_sl))
                                    self.risk_reward_var.set(str(rec_rr))
                                    self.max_spread_var.set(str(rec_spread))
                                    self.use_ml_var.set(True)
                                    
                                    self.ml_log_message("✓ Recommended settings applied!", "SUCCESS")
                                    self.log_message("✓ Configuration updated with recommended settings", "SUCCESS")
                                else:
                                    self.ml_log_message("ℹ Keeping current settings", "INFO")
                            else:
                                messagebox.showinfo("Success", f"ML models loaded!\\n\\nSymbol: {selected_symbol[0]}\\nTest Accuracy: {test_acc:.2f}%")
                        
                        except Exception as e:
                            self.ml_log_message(f"⚠ Metadata not found, using defaults", "WARNING")
                            messagebox.showinfo("Success", f"ML models loaded!\\n\\nSymbol: {selected_symbol[0]}\\nFolder: {symbol_folder}")
                    else:
                        self.ml_log_message(f"⚠ No metadata found (older model)", "WARNING")
                        messagebox.showinfo("Success", f"ML models loaded!\\n\\nSymbol: {selected_symbol[0]}\\nFolder: {symbol_folder}")
                    
                    self.use_ml_var.set(True)
                    self.update_ml_status()
                else:
                    self.log_message("Failed to load models", "ERROR")
                    self.ml_log_message("✗ Failed to load models", "ERROR")
                    messagebox.showerror("Error", "Failed to load models. Check if model files are complete.")
            
        except Exception as e:
            self.log_message(f"Error loading models: {str(e)}", "ERROR")
            self.ml_log_message(f"✗ Error: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
    
    def update_ml_status(self):
        """Update ML status display"""
        try:
            if not hasattr(self, 'ml_info_text'):
                return
            
            self.ml_info_text.delete(1.0, tk.END)
            
            status_text = self.t('ml_status_title') + "\n"
            status_text += "=" * 60 + "\n\n"
            
            # ML Enabled Status
            ml_enabled = self.use_ml_var.get()
            status_emoji = "🟢" if ml_enabled else "🔴"
            ml_status_text = self.t('enabled') if ml_enabled else self.t('disabled')
            status_text += f"{status_emoji} {self.t('ml_predictions_label')} {ml_status_text}\n\n"
            
            # Model Status
            if self.ml_predictor:
                status_text += self.t('model_info_title') + "\n"
                status_text += "-" * 60 + "\n"
                status_text += f"{self.t('current_symbol')} {self.symbol_var.get()}\n"
                status_text += self.t('models_loaded') + "\n"
                status_text += self.t('model_type') + "\n"
                status_text += self.t('features') + "\n"
                status_text += self.t('prediction_window') + "\n\n"
                
                # Model Architecture
                status_text += self.t('model_arch_title') + "\n"
                status_text += "-" * 60 + "\n"
                status_text += self.t('lstm_desc') + "\n"
                status_text += self.t('lstm_seq') + "\n"
                status_text += self.t('lstm_units') + "\n"
                status_text += self.t('lstm_purpose') + "\n\n"
                
                status_text += self.t('gru_desc') + "\n"
                status_text += self.t('gru_seq') + "\n"
                status_text += self.t('gru_units') + "\n"
                status_text += self.t('gru_purpose') + "\n\n"
                
                status_text += self.t('cnn_desc') + "\n"
                status_text += self.t('cnn_filters') + "\n"
                status_text += self.t('cnn_kernel') + "\n"
                status_text += self.t('cnn_purpose') + "\n\n"
                
                status_text += self.t('ensemble_desc') + "\n"
                status_text += self.t('ensemble_combines') + "\n"
                status_text += self.t('ensemble_threshold') + "\n\n"
                
                # Training Info
                import os
                current_symbol = self.symbol_var.get()
                symbol_model_path = f"./models/{current_symbol}"
                
                status_text += self.t('training_info_title') + "\n"
                status_text += "-" * 60 + "\n"
                
                # Check if current symbol has models
                if os.path.exists(symbol_model_path):
                    # Check model files in symbol folder
                    model_files = []
                    if os.path.exists(f"{symbol_model_path}/lstm_model.h5"):
                        model_files.append("LSTM")
                    if os.path.exists(f"{symbol_model_path}/gru_model.h5"):
                        model_files.append("GRU")
                    if os.path.exists(f"{symbol_model_path}/cnn_model.h5"):
                        model_files.append("CNN")
                    if os.path.exists(f"{symbol_model_path}/direction_model.pkl"):
                        model_files.append("Direction")
                    if os.path.exists(f"{symbol_model_path}/confidence_model.pkl"):
                        model_files.append("Confidence")
                    
                    if model_files:
                        status_text += f"{self.t('current_symbol')} {current_symbol}\n"
                        status_text += f"{self.t('model_location')} {symbol_model_path}/\n"
                        status_text += f"{self.t('available_models')} {', '.join(model_files)}\n"
                        
                        # Get file modification time as last training time
                        try:
                            import time
                            for model_file in ['lstm_model.h5', 'gru_model.h5', 'cnn_model.h5', 
                                             'direction_model.pkl', 'confidence_model.pkl']:
                                model_file_path = f"{symbol_model_path}/{model_file}"
                                if os.path.exists(model_file_path):
                                    mtime = os.path.getmtime(model_file_path)
                                    last_train = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                                    status_text += f"{self.t('last_trained')} {last_train}\n"
                                    break
                        except:
                            status_text += f"{self.t('last_trained')} Unknown\n"
                        
                        status_text += f"{self.t('training_days')} {self.ml_days_var.get()}\n"
                        status_text += self.t('status_ready') + "\n\n"
                else:
                    status_text += f"{self.t('current_symbol')} {current_symbol}\n"
                    status_text += f"{self.t('no_trained_models')} {current_symbol}\n"
                    status_text += f"{self.t('expected_location')} {symbol_model_path}/\n\n"
                
                # List all available symbol models
                models_dir = "./models"
                if os.path.exists(models_dir):
                    available_symbols = []
                    for item in os.listdir(models_dir):
                        item_path = os.path.join(models_dir, item)
                        if os.path.isdir(item_path):
                            has_models = any(
                                os.path.exists(os.path.join(item_path, f))
                                for f in ['lstm_model.h5', 'gru_model.h5', 'cnn_model.h5',
                                         'direction_model.pkl', 'confidence_model.pkl']
                            )
                            if has_models:
                                available_symbols.append(item)
                    
                    if available_symbols:
                        status_text += self.t('all_available_models') + "\n"
                        status_text += "-" * 60 + "\n"
                        for sym in sorted(available_symbols):
                            indicator = "✓" if sym == current_symbol else "○"
                            status_text += f"{indicator} {sym}\n"
                        status_text += "\n"
                        status_text += self.t('use_load_models') + "\n\n"
                
                # Usage Guide
                status_text += self.t('usage_guide_title') + "\n"
                status_text += "-" * 60 + "\n"
                status_text += self.t('usage_train') + "\n"
                status_text += self.t('usage_train_1') + "\n"
                status_text += self.t('usage_train_2') + "\n"
                status_text += self.t('usage_train_3') + "\n\n"
                
                status_text += self.t('usage_use') + "\n"
                status_text += self.t('usage_use_1') + "\n"
                status_text += self.t('usage_use_2') + "\n"
                status_text += self.t('usage_use_3') + "\n\n"
                
                status_text += self.t('usage_manage') + "\n"
                status_text += self.t('usage_manage_1') + "\n"
                status_text += self.t('usage_manage_2') + "\n"
                status_text += self.t('usage_manage_3') + "\n\n"
                
            else:
                status_text += self.t('no_models_loaded') + "\n"
                status_text += "-" * 60 + "\n\n"
                status_text += self.t('to_use_ml') + "\n"
                status_text += self.t('train_new_models') + "\n"
                status_text += self.t('or_text') + "\n"
                status_text += self.t('load_existing_models') + "\n\n"
                
                status_text += self.t('training_info_text') + "\n\n"
            
            # Performance Tips
            status_text += self.t('performance_tips_title') + "\n"
            status_text += "-" * 60 + "\n"
            status_text += self.t('tip_more_data') + "\n"
            status_text += self.t('tip_retrain') + "\n"
            status_text += self.t('tip_signal_strength') + "\n"
            status_text += self.t('tip_combine') + "\n"
            status_text += self.t('tip_monitor') + "\n\n"
            
            self.ml_info_text.insert(1.0, status_text)
            
            # Initial log message if empty
            if self.ml_status_text.get(1.0, tk.END).strip() == "":
                self.ml_log_message(self.t('ml_initialized'), "INFO")
                
        except Exception as e:
            self.log_message(f"Error updating ML status: {str(e)}", "ERROR")
    
    def ml_log_message(self, message: str, level: str = "INFO"):
        """Add message to ML logs - thread safe"""
        # Check if widget exists and is valid
        if not hasattr(self, 'ml_status_text'):
            # Widget not created yet, just log to main log
            self.log_message(f"[ML] {message}", level)
            return
        
        try:
            # Use root.after to ensure GUI updates happen in main thread
            def update_log():
                try:
                    if not hasattr(self, 'ml_status_text'):
                        return
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"[{timestamp}] {level}: {message}\n"
                    
                    self.ml_status_text.insert(tk.END, log_entry)
                    self.ml_status_text.see(tk.END)
                    
                    # Color coding
                    if level == "ERROR":
                        self.ml_status_text.tag_add("error", f"{self.ml_status_text.index(tk.END)}-2l", f"{self.ml_status_text.index(tk.END)}-1l")
                        self.ml_status_text.tag_config("error", foreground="#ff1744", font=('Consolas', 9, 'bold'))
                    elif level == "SUCCESS":
                        self.ml_status_text.tag_add("success", f"{self.ml_status_text.index(tk.END)}-2l", f"{self.ml_status_text.index(tk.END)}-1l")
                        self.ml_status_text.tag_config("success", foreground="#00e676", font=('Consolas', 9, 'bold'))
                except:
                    pass  # Widget was destroyed
            
            # Schedule update in main thread
            self.root.after(0, update_log)
        except:
            # If root.after fails, just log to main log
            self.log_message(f"[ML] {message}", level)
    
    def update_risk_status(self):
        """Update risk management status display"""
        try:
            if not hasattr(self, 'risk_metrics_text'):
                return
            
            self.risk_metrics_text.delete(1.0, tk.END)
            
            status_text = "🛡️ RISK MANAGEMENT STATUS\n"
            status_text += "=" * 60 + "\n\n"
            
            # Get current limits
            max_daily_loss = float(self.limit_vars['max_daily_loss'].get())
            max_daily_trades = int(self.limit_vars['max_daily_trades'].get())
            max_position_size = float(self.limit_vars['max_position_size'].get())
            max_positions = int(self.limit_vars['max_positions'].get())
            max_drawdown = float(self.limit_vars['max_drawdown_pct'].get())
            
            # Risk Manager Status
            if self.risk_manager:
                status_text += "✅ RISK MANAGER: ACTIVE\n"
                status_text += "-" * 60 + "\n\n"
                
                try:
                    summary = self.risk_manager.get_trading_summary()
                    
                    # Current Status
                    status_text += "📊 CURRENT STATUS:\n"
                    status_text += "-" * 60 + "\n"
                    
                    # Trading enabled/disabled
                    if summary.get('circuit_breaker_triggered', False):
                        status_text += "🚨 STATUS: CIRCUIT BREAKER ACTIVE!\n"
                        status_text += "   Trading is HALTED due to risk limits\n\n"
                    elif summary.get('trading_enabled', True):
                        status_text += "🟢 STATUS: Trading ENABLED\n"
                        status_text += "   All systems operational\n\n"
                    else:
                        status_text += "🔴 STATUS: Trading DISABLED\n"
                        status_text += "   Check risk limits\n\n"
                    
                    # Daily Metrics
                    status_text += "📅 TODAY'S METRICS:\n"
                    status_text += "-" * 60 + "\n"
                    
                    daily_pnl = summary.get('daily_pnl', 0.0)
                    daily_trades = summary.get('daily_trades', 0)
                    current_drawdown = summary.get('current_drawdown', 0.0)
                    
                    # Daily PnL with color indicator
                    pnl_indicator = "🟢" if daily_pnl >= 0 else "🔴"
                    status_text += f"{pnl_indicator} Daily P/L: ${daily_pnl:.2f}\n"
                    status_text += f"   Limit: ${max_daily_loss:.2f}\n"
                    
                    if daily_pnl < 0:
                        usage_pct = abs(daily_pnl) / max_daily_loss * 100 if max_daily_loss > 0 else 0
                        status_text += f"   Usage: {usage_pct:.1f}% of limit\n"
                    status_text += "\n"
                    
                    # Daily Trades
                    status_text += f"📈 Daily Trades: {daily_trades} / {max_daily_trades}\n"
                    if max_daily_trades > 0:
                        trades_pct = (daily_trades / max_daily_trades) * 100
                        status_text += f"   Usage: {trades_pct:.1f}% of limit\n"
                    status_text += "\n"
                    
                    # Drawdown
                    dd_indicator = "🟢" if current_drawdown < max_drawdown * 0.5 else "🟡" if current_drawdown < max_drawdown * 0.8 else "🔴"
                    status_text += f"{dd_indicator} Drawdown: {current_drawdown:.2f}%\n"
                    status_text += f"   Limit: {max_drawdown:.2f}%\n"
                    if max_drawdown > 0:
                        dd_usage = (current_drawdown / max_drawdown) * 100
                        status_text += f"   Usage: {dd_usage:.1f}% of limit\n"
                    status_text += "\n"
                    
                    # Overall Statistics
                    status_text += "📊 OVERALL STATISTICS:\n"
                    status_text += "-" * 60 + "\n"
                    
                    total_trades = summary.get('total_trades', 0)
                    wins = summary.get('wins', 0)
                    losses = summary.get('losses', 0)
                    win_rate = summary.get('win_rate', 0.0)
                    
                    status_text += f"Total Trades: {total_trades}\n"
                    status_text += f"Wins: {wins} | Losses: {losses}\n"
                    status_text += f"Win Rate: {win_rate*100:.1f}%\n\n"
                    
                    total_profit = summary.get('total_profit', 0.0)
                    total_loss = summary.get('total_loss', 0.0)
                    net_profit = summary.get('net_profit', 0.0)
                    profit_factor = summary.get('profit_factor', 0.0)
                    
                    status_text += f"Total Profit: ${total_profit:.2f}\n"
                    status_text += f"Total Loss: ${total_loss:.2f}\n"
                    net_indicator = "🟢" if net_profit >= 0 else "🔴"
                    status_text += f"{net_indicator} Net Profit: ${net_profit:.2f}\n"
                    status_text += f"Profit Factor: {profit_factor:.2f}\n\n"
                    
                except Exception as e:
                    status_text += f"⚠️ Could not retrieve risk metrics: {str(e)}\n\n"
                
            else:
                status_text += "⚠️ RISK MANAGER: NOT INITIALIZED\n"
                status_text += "-" * 60 + "\n\n"
                status_text += "Risk manager will be initialized when trading starts.\n\n"
            
            # Current Risk Limits
            status_text += "⚙️ CONFIGURED LIMITS:\n"
            status_text += "-" * 60 + "\n"
            status_text += f"Max Daily Loss: ${max_daily_loss:.2f}\n"
            status_text += f"Max Daily Trades: {max_daily_trades}\n"
            status_text += f"Max Position Size: {max_position_size}\n"
            status_text += f"Max Positions: {max_positions}\n"
            status_text += f"Max Drawdown: {max_drawdown:.2f}%\n\n"
            
            # Current Positions (if MT5 available)
            if mt5.initialize():
                positions = mt5.positions_get()
                if positions:
                    status_text += "📍 CURRENT POSITIONS:\n"
                    status_text += "-" * 60 + "\n"
                    status_text += f"Open Positions: {len(positions)} / {max_positions}\n"
                    
                    total_floating_pnl = sum([pos.profit for pos in positions])
                    pnl_emoji = "🟢" if total_floating_pnl >= 0 else "🔴"
                    status_text += f"{pnl_emoji} Floating P/L: ${total_floating_pnl:.2f}\n"
                    
                    total_volume = sum([pos.volume for pos in positions])
                    status_text += f"Total Volume: {total_volume:.2f}\n"
                    
                    if total_volume > max_position_size * max_positions:
                        status_text += "⚠️ WARNING: Total volume exceeds safe limits!\n"
                    
                    status_text += "\n"
                else:
                    status_text += "📍 CURRENT POSITIONS: None\n\n"
                mt5.shutdown()
            
            # Risk Warnings
            status_text += "⚠️ RISK WARNINGS:\n"
            status_text += "-" * 60 + "\n"
            
            warnings = []
            
            if self.risk_manager:
                try:
                    summary = self.risk_manager.get_trading_summary()
                    
                    # Check daily loss
                    if abs(summary.get('daily_pnl', 0)) > max_daily_loss * 0.8:
                        warnings.append("⚠️ Daily loss approaching limit (>80%)")
                    
                    # Check daily trades
                    if summary.get('daily_trades', 0) > max_daily_trades * 0.8:
                        warnings.append("⚠️ Daily trades approaching limit (>80%)")
                    
                    # Check drawdown
                    if summary.get('current_drawdown', 0) > max_drawdown * 0.8:
                        warnings.append("⚠️ Drawdown approaching limit (>80%)")
                    
                    # Check circuit breaker
                    if summary.get('circuit_breaker_triggered', False):
                        warnings.append("🚨 CIRCUIT BREAKER ACTIVE - Trading halted!")
                    
                    # Check win rate
                    if summary.get('total_trades', 0) > 10:
                        if summary.get('win_rate', 1.0) < 0.3:
                            warnings.append("⚠️ Low win rate (<30%) - Review strategy")
                    
                except:
                    pass
            
            if warnings:
                for warning in warnings:
                    status_text += f"{warning}\n"
            else:
                status_text += "✅ No active warnings\n"
            
            status_text += "\n"
            
            # Best Practices
            status_text += "💡 RISK MANAGEMENT BEST PRACTICES:\n"
            status_text += "-" * 60 + "\n"
            status_text += "• Never risk more than 1-2% per trade\n"
            status_text += "• Keep max drawdown under 10-15%\n"
            status_text += "• Monitor daily P/L regularly\n"
            status_text += "• Set realistic profit targets\n"
            status_text += "• Use stop losses on every trade\n"
            status_text += "• Review and adjust limits weekly\n"
            status_text += "• Circuit breaker is your safety net\n\n"
            
            self.risk_metrics_text.insert(1.0, status_text)
            
            # Initial log message if empty
            if self.risk_status_text.get(1.0, tk.END).strip() == "":
                self.risk_log_message("Risk management system initialized", "INFO")
                self.risk_log_message(f"Max daily loss set to ${max_daily_loss:.2f}", "INFO")
                self.risk_log_message(f"Max daily trades set to {max_daily_trades}", "INFO")
                self.risk_log_message(f"Max drawdown set to {max_drawdown:.2f}%", "INFO")
                
        except Exception as e:
            self.log_message(f"Error updating risk status: {str(e)}", "ERROR")
    
    def risk_log_message(self, message: str, level: str = "INFO"):
        """Add message to risk logs - thread safe"""
        # Check if widget exists and is valid
        if not hasattr(self, 'risk_status_text'):
            # Widget not created yet, just log to main log
            self.log_message(f"[Risk] {message}", level)
            return
        
        try:
            # Use root.after to ensure GUI updates happen in main thread
            def update_log():
                try:
                    if not hasattr(self, 'risk_status_text'):
                        return
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"[{timestamp}] {level}: {message}\n"
                    
                    self.risk_status_text.insert(tk.END, log_entry)
                    self.risk_status_text.see(tk.END)
                    
                    # Color coding
                    if level == "ERROR" or level == "WARNING":
                        self.risk_status_text.tag_add("error", f"{self.risk_status_text.index(tk.END)}-2l", f"{self.risk_status_text.index(tk.END)}-1l")
                        self.risk_status_text.tag_config("error", foreground="#f48771")
                    elif level == "SUCCESS":
                        self.risk_status_text.tag_add("success", f"{self.risk_status_text.index(tk.END)}-2l", f"{self.risk_status_text.index(tk.END)}-1l")
                        self.risk_status_text.tag_config("success", foreground="#4ec9b0")
                    elif level == "ALERT":
                        self.risk_status_text.tag_add("alert", f"{self.risk_status_text.index(tk.END)}-2l", f"{self.risk_status_text.index(tk.END)}-1l")
                        self.risk_status_text.tag_config("alert", foreground="#ffcc00")
                except:
                    pass  # Widget was destroyed
            
            # Schedule update in main thread
            self.root.after(0, update_log)
        except:
            # If root.after fails, just log to main log
            self.log_message(f"[Risk] {message}", level)
    
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
    
    def reset_chart(self):
        """Reset chart data"""
        self.chart_data = {
            'timestamps': deque(maxlen=300),
            'pnl': deque(maxlen=300),
            'equity': deque(maxlen=300),
            'trades': deque(maxlen=300),
            'win_rate': deque(maxlen=300),
            'latency': deque(maxlen=300)
        }
        
        # Clear plot lines
        self.line_pnl.set_data([], [])
        self.line_winrate.set_data([], [])
        self.line_trades.set_data([], [])
        self.line_latency.set_data([], [])
        
        self.canvas.draw()
        self.log_message("Chart data reset", "INFO")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            # Ask user for filename
            filename = filedialog.asksaveasfilename(
                title="Save Configuration",
                initialdir=".",
                initialfile="hft_config.json",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if not filename:  # User cancelled
                return
            
            config = {
                'symbol': self.symbol_var.get(),
                'volume': self.volume_var.get(),
                'magic': self.magic_var.get(),
                'risk_per_trade': self.risk_var.get(),
                'min_signal_strength': self.signal_strength_var.get(),
                'max_spread': self.max_spread_var.get(),
                'max_volatility': self.max_volatility_var.get(),
                'filling_mode': self.filling_mode_var.get(),
                'sl_multiplier': self.sl_multiplier_var.get(),
                'risk_reward_ratio': self.risk_reward_var.get(),
                'tp_mode': self.tp_mode_var.get(),
                'tp_dollar_amount': self.tp_dollar_var.get(),
                'max_floating_loss': self.max_floating_loss_var.get(),
                'max_floating_profit': self.max_floating_profit_var.get(),
                'mt5_path': self.mt5_path_var.get(),
                'use_ml': self.use_ml_var.get(),
                'limits': {k: v.get() for k, v in self.limit_vars.items()}
            }
            
            with open(filename, 'w') as f:
                json.dump(config, f, indent=4)
            
            self.log_message(f"✓ Configuration saved to {filename}", "SUCCESS")
            messagebox.showinfo("Success", f"Configuration saved to:\n{filename}")
            
        except Exception as e:
            self.log_message(f"Error saving config: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Failed to save configuration:\n{str(e)}")
    
    def load_config(self):
        """Load configuration from file"""
        try:
            # Ask user to select file
            filename = filedialog.askopenfilename(
                title="Load Configuration",
                initialdir=".",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if not filename:  # User cancelled
                return
            
            with open(filename, 'r') as f:
                config = json.load(f)
            
            # Load basic settings
            self.symbol_var.set(config.get('symbol', 'GOLD.ls'))
            self.volume_var.set(config.get('volume', '0.01'))
            self.magic_var.set(config.get('magic', '2026001'))
            self.risk_var.set(config.get('risk_per_trade', '1.0'))
            self.signal_strength_var.set(config.get('min_signal_strength', '0.4'))
            
            # Load spread and volatility if available
            if hasattr(self, 'max_spread_var'):
                self.max_spread_var.set(config.get('max_spread', '0.05'))
            if hasattr(self, 'max_volatility_var'):
                self.max_volatility_var.set(config.get('max_volatility', '0.1'))
            if hasattr(self, 'filling_mode_var'):
                self.filling_mode_var.set(config.get('filling_mode', 'FOK'))
            if hasattr(self, 'sl_multiplier_var'):
                self.sl_multiplier_var.set(config.get('sl_multiplier', '2.0'))
            if hasattr(self, 'risk_reward_var'):
                self.risk_reward_var.set(config.get('risk_reward_ratio', '2.0'))
            if hasattr(self, 'tp_mode_var'):
                self.tp_mode_var.set(config.get('tp_mode', 'RiskReward'))
            if hasattr(self, 'tp_dollar_var'):
                self.tp_dollar_var.set(config.get('tp_dollar_amount', '0.5'))
            if hasattr(self, 'max_floating_loss_var'):
                self.max_floating_loss_var.set(config.get('max_floating_loss', '500'))
            if hasattr(self, 'max_floating_profit_var'):
                self.max_floating_profit_var.set(config.get('max_floating_profit', '1'))
            if hasattr(self, 'mt5_path_var'):
                self.mt5_path_var.set(config.get('mt5_path', 'C:\\Program Files\\XM Global MT5\\terminal64.exe'))
            
            self.use_ml_var.set(config.get('use_ml', False))
            
            # Load risk limits
            if 'limits' in config:
                for k, v in config['limits'].items():
                    if k in self.limit_vars:
                        self.limit_vars[k].set(v)
            
            self.log_message(f"✓ Configuration loaded from {filename}", "SUCCESS")
            messagebox.showinfo("Success", f"Configuration loaded from:\n{filename}\n\nSymbol: {config.get('symbol')}\nMax Spread: {config.get('max_spread', 'N/A')}\nMax Volatility: {config.get('max_volatility', 'N/A')}")
            
        except FileNotFoundError:
            self.log_message("Config file not found", "ERROR")
            messagebox.showerror("Error", "Configuration file not found!")
        except json.JSONDecodeError:
            self.log_message("Invalid JSON in config file", "ERROR")
            messagebox.showerror("Error", "Invalid configuration file format!")
        except Exception as e:
            self.log_message(f"Error loading config: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Failed to load configuration:\n{str(e)}")
    
    def quick_load_config(self, filename):
        """Quick load preset configuration"""
        try:
            import os
            
            # Check if file exists
            if not os.path.exists(filename):
                self.log_message(f"Preset file not found: {filename}", "ERROR")
                messagebox.showerror("Error", f"Preset configuration not found:\n{filename}\n\nMake sure the preset files are in the same directory as the application.")
                return
            
            with open(filename, 'r') as f:
                config = json.load(f)
            
            # Load from JSON structure (may have nested structure)
            if 'spread_settings' in config:
                # New JSON format with nested structure
                self.symbol_var.set(config.get('symbol', 'GOLD.ls'))
                self.volume_var.set(str(config.get('default_volume', 0.01)))
                self.magic_var.set(str(config.get('magic_number', 2026001)))
                
                # Load from nested structures
                spread_settings = config.get('spread_settings', {})
                self.max_spread_var.set(str(spread_settings.get('max_spread', 0.05)))
                
                signal_settings = config.get('signal_settings', {})
                self.signal_strength_var.set(str(signal_settings.get('min_signal_strength', 0.4)))
                
                volatility_settings = config.get('volatility_settings', {})
                self.max_volatility_var.set(str(volatility_settings.get('max_volatility', 0.1)))
                
                execution_settings = config.get('execution_settings', {})
                if 'filling_mode' in execution_settings:
                    filling_map = {'FOK': 'FOK', 'IOC': 'IOC', 'RETURN': 'RETURN'}
                    self.filling_mode_var.set(filling_map.get(execution_settings.get('filling_mode', 'FOK'), 'FOK'))
                
                risk_settings = config.get('risk_settings', {})
                self.risk_var.set(str(risk_settings.get('risk_per_trade', 0.01) * 100))
                
                # Load SL/TP settings
                if 'sl_multiplier' in execution_settings:
                    self.sl_multiplier_var.set(str(execution_settings.get('sl_multiplier', 2.0)))
                if 'risk_reward_ratio' in risk_settings:
                    self.risk_reward_var.set(str(risk_settings.get('risk_reward_ratio', 2.0)))
                if 'tp_mode' in execution_settings:
                    self.tp_mode_var.set(execution_settings.get('tp_mode', 'RiskReward'))
                if 'tp_dollar_amount' in execution_settings:
                    self.tp_dollar_var.set(str(execution_settings.get('tp_dollar_amount', 0.5)))
                if 'max_floating_loss' in risk_settings:
                    self.max_floating_loss_var.set(str(risk_settings.get('max_floating_loss', 500)))
                if 'max_floating_profit' in risk_settings:
                    self.max_floating_profit_var.set(str(risk_settings.get('max_floating_profit', 1)))
                
                # Load MT5 path
                if 'mt5_path' in execution_settings:
                    self.mt5_path_var.set(execution_settings.get('mt5_path', 'C:\\Program Files\\XM Global MT5\\terminal64.exe'))
                elif 'mt5_path' in config:
                    self.mt5_path_var.set(config.get('mt5_path', 'C:\\Program Files\\XM Global MT5\\terminal64.exe'))
                
                # Load risk limits if available
                if 'max_daily_loss' in risk_settings:
                    self.limit_vars['max_daily_loss'].set(str(risk_settings['max_daily_loss']))
                if 'max_daily_trades' in risk_settings:
                    self.limit_vars['max_daily_trades'].set(str(risk_settings['max_daily_trades']))
                if 'max_position_size' in risk_settings:
                    self.limit_vars['max_position_size'].set(str(risk_settings['max_position_size']))
                if 'max_positions' in risk_settings:
                    self.limit_vars['max_positions'].set(str(risk_settings['max_positions']))
                if 'max_drawdown_pct' in risk_settings:
                    self.limit_vars['max_drawdown_pct'].set(str(risk_settings['max_drawdown_pct']))
            else:
                # Simple flat format (backward compatibility)
                self.symbol_var.set(config.get('symbol', 'GOLD.ls'))
                self.volume_var.set(config.get('volume', '0.01'))
                self.magic_var.set(config.get('magic', '2026001'))
                self.risk_var.set(config.get('risk_per_trade', '1.0'))
                self.signal_strength_var.set(config.get('min_signal_strength', '0.4'))
                self.max_spread_var.set(config.get('max_spread', '0.05'))
                self.max_volatility_var.set(config.get('max_volatility', '0.1'))
                if 'filling_mode' in config:
                    self.filling_mode_var.set(config.get('filling_mode', 'FOK'))
                if 'sl_multiplier' in config:
                    self.sl_multiplier_var.set(config.get('sl_multiplier', '2.0'))
                if 'risk_reward_ratio' in config:
                    self.risk_reward_var.set(config.get('risk_reward_ratio', '2.0'))
                if 'tp_mode' in config:
                    self.tp_mode_var.set(config.get('tp_mode', 'RiskReward'))
                if 'tp_dollar_amount' in config:
                    self.tp_dollar_var.set(config.get('tp_dollar_amount', '0.5'))
                if 'max_floating_loss' in config:
                    self.max_floating_loss_var.set(config.get('max_floating_loss', '500'))
                if 'max_floating_profit' in config:
                    self.max_floating_profit_var.set(config.get('max_floating_profit', '1'))
                if 'mt5_path' in config:
                    self.mt5_path_var.set(config.get('mt5_path', 'C:\\Program Files\\XM Global MT5\\terminal64.exe'))
                
                if 'limits' in config:
                    for k, v in config['limits'].items():
                        if k in self.limit_vars:
                            self.limit_vars[k].set(v)
            
            self.use_ml_var.set(False)
            
            preset_name = filename.replace('config_', '').replace('.json', '')
            self.log_message(f"✓ Loaded preset: {preset_name}", "SUCCESS")
            messagebox.showinfo("Success", f"Preset configuration loaded: {preset_name}\n\nSymbol: {self.symbol_var.get()}\nMax Spread: {self.max_spread_var.get()}\nMax Volatility: {self.max_volatility_var.get()}")
            
        except FileNotFoundError:
            self.log_message(f"Preset not found: {filename}", "ERROR")
            messagebox.showerror("Error", f"Preset file not found:\n{filename}")
        except json.JSONDecodeError:
            self.log_message("Invalid preset file format", "ERROR")
            messagebox.showerror("Error", "Invalid configuration file format!")
        except Exception as e:
            self.log_message(f"Error loading preset: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Failed to load preset:\n{str(e)}")
    
    def browse_mt5_path(self):
        """Browse for MT5 terminal executable"""
        from tkinter import filedialog
        
        filename = filedialog.askopenfilename(
            title="Select MT5 Terminal Executable",
            initialdir="C:\\Program Files",
            filetypes=[("Executable files", "*.exe"), ("All files", "*.*")]
        )
        
        if filename:
            self.mt5_path_var.set(filename)
            self.log_message(f"MT5 path set to: {filename}", "INFO")
    
    def export_logs(self):
        """Export logs to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hft_pro_logs_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            
            self.log_message(f"✓ Logs exported to {filename}", "SUCCESS")
            messagebox.showinfo("Success", f"Logs exported to {filename}")
            
        except Exception as e:
            self.log_message(f"Error exporting logs: {str(e)}", "ERROR")
    
    def export_status_logs(self):
        """Export status logs to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hft_status_logs_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(self.status_text.get(1.0, tk.END))
            
            self.log_message(f"✓ Status logs exported to {filename}", "SUCCESS")
            messagebox.showinfo("Success", f"Status logs exported to {filename}")
            
        except Exception as e:
            self.log_message(f"Error exporting status logs: {str(e)}", "ERROR")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            if messagebox.askokcancel("Quit", "Trading is active. Do you want to stop and quit?"):
                self.stop_trading()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = HFTProGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
