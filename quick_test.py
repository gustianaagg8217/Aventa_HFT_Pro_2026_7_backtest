"""
Quick Test - Aventa HFT Pro 2026
Test trading engine with more aggressive settings to verify signal generation
"""

import time
import logging
from aventa_hft_core import UltraLowLatencyEngine

# Configure logging to see all details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# More aggressive config for testing
config = {
    'magic_number': 2026999,  # Different magic for testing
    'default_volume': 0.01,
    
    # LOWER THRESHOLDS FOR MORE SIGNALS (TESTING ONLY!)
    'min_delta_threshold': 5,          # Very low for testing
    'min_velocity_threshold': 0.0001,  # Adjusted for Gold movement (larger)
    'max_spread': 0.05,                # GOLD: 2pt spread × 0.01 tick = 0.02, set to 0.05 for safety
    'max_volatility': 0.1,             # Gold is more volatile
    'min_signal_strength': 0.3,        # Lower from 0.6 -> MORE SIGNALS! 🚀
    
    'risk_reward_ratio': 2.0,
    'analysis_interval': 0.5,          # 500ms - analyze more frequently
    'slippage': 50                     # Higher slippage tolerance
}

logger.info("=" * 70)
logger.info("🧪 QUICK TEST MODE - More aggressive settings for signal testing")
logger.info("=" * 70)
logger.info("")
logger.info("⚙️ TEST CONFIGURATION:")
logger.info(f"   • Min Signal Strength: {config['min_signal_strength']} (LOWER = MORE SIGNALS)")
logger.info(f"   • Min Delta Threshold: {config['min_delta_threshold']}")
logger.info(f"   • Min Velocity: {config['min_velocity_threshold']} (Adjusted for GOLD)")
logger.info(f"   • Max Spread: {config['max_spread']} (GOLD: 2pt × 0.01 tick = 0.02, allow up to 0.05)")
logger.info(f"   • Max Volatility: {config['max_volatility']} (GOLD is more volatile)")
logger.info(f"   • Analysis Interval: {config['analysis_interval']}s")
logger.info("")
logger.info("⚠️ WARNING: This config is for TESTING ONLY!")
logger.info("   DO NOT use this in live trading without proper backtesting!")
logger.info("")

# Ask user for symbol
print("Enter symbol to test (e.g., EURUSD, GOLD.ls, XAUUSD):")
print("Or press ENTER for default: GOLD.ls")
symbol_input = input("> ").strip().upper()

if not symbol_input:
    symbol_input = "GOLD.ls"

logger.info(f"Testing with symbol: {symbol_input}")
logger.info("")

# Create engine
engine = UltraLowLatencyEngine(symbol=symbol_input, config=config)

try:
    # Start engine
    if engine.start():
        logger.info("")
        logger.info("✅ Engine started! Monitoring for 2 minutes...")
        logger.info("   Watch for: 📊 SIGNAL GENERATED messages")
        logger.info("   Press Ctrl+C to stop early")
        logger.info("")
        
        test_duration = 120  # 2 minutes
        elapsed = 0
        
        while elapsed < test_duration:
            time.sleep(10)
            elapsed += 10
            
            # Print performance stats
            stats = engine.get_performance_stats()
            logger.info("")
            logger.info(f"⏱️ [{elapsed}/{test_duration}s] Performance Stats:")
            logger.info(f"   • Latency: {stats['tick_latency_avg_us']:.1f}μs")
            logger.info(f"   • Exec Time: {stats['execution_time_avg_ms']:.2f}ms")
            logger.info(f"   • Ticks Processed: {stats['ticks_processed']:,}")
            logger.info(f"   • Order Flow Samples: {stats['orderflow_samples']}")
            logger.info(f"   • Current Position: {stats['current_position'] or 'None'}")
            logger.info("")
        
        logger.info("=" * 70)
        logger.info("🏁 Test completed!")
        logger.info("")
        
        # Final stats
        stats = engine.get_performance_stats()
        logger.info("📊 FINAL STATISTICS:")
        logger.info(f"   • Total Ticks: {stats['ticks_processed']:,}")
        logger.info(f"   • Avg Latency: {stats['tick_latency_avg_us']:.1f}μs")
        logger.info(f"   • Min Latency: {stats['tick_latency_min_us']:.1f}μs")
        logger.info(f"   • Max Latency: {stats['tick_latency_max_us']:.1f}μs")
        
        if stats['current_position']:
            logger.info(f"   • Open Position: {stats['current_position']} ({stats['position_volume']} lots)")
        
        logger.info("")
        logger.info("💡 ANALYSIS:")
        
        if stats['ticks_processed'] == 0:
            logger.warning("   ⚠️ NO TICKS PROCESSED - Check MT5 connection & symbol availability")
        elif stats['orderflow_samples'] == 0:
            logger.warning("   ⚠️ NO ORDER FLOW DATA - Ticks not changing (market closed?)")
        elif stats['current_position'] is None:
            logger.info("   ℹ️ No position opened during test period")
            logger.info("   Possible reasons:")
            logger.info("      • Market conditions didn't meet signal criteria")
            logger.info("      • Not enough volatility/momentum")
            logger.info("      • Spread too wide")
            logger.info("      • Check logs above for 'Weak signal' messages")
        else:
            logger.info("   ✅ Position opened successfully!")
        
        logger.info("")
        
except KeyboardInterrupt:
    logger.info("")
    logger.info("⏹️ Test stopped by user")
except Exception as e:
    logger.error(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()
finally:
    logger.info("")
    logger.info("🔄 Shutting down engine...")
    engine.stop()
    logger.info("✅ Shutdown complete")
    logger.info("=" * 70)
