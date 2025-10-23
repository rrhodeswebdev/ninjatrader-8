"""
Risk Management Module for Trading System

Provides position sizing, stop loss, and take profit calculations
based on confidence, volatility, and market regime.
"""

import numpy as np
from typing import Tuple, Dict, Optional


class PositionSizer:
    """
    Calculate position sizes using Kelly Criterion and confidence-based scaling
    """

    def __init__(
        self,
        base_risk_pct: float = 0.01,  # Risk 1% of account per trade (REDUCED from 2%)
        max_risk_pct: float = 0.02,   # Maximum 2% risk (REDUCED from 3%)
        min_risk_pct: float = 0.005,  # Minimum 0.5% risk
        confidence_threshold: float = 0.65,  # Minimum confidence to trade
        max_contracts: int = 10  # Maximum position size (safety limit)
    ):
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.min_risk_pct = min_risk_pct
        self.confidence_threshold = confidence_threshold
        self.max_contracts = max_contracts

    def calculate_position_size(
        self,
        confidence: float,
        account_balance: float,
        stop_loss_distance: float,
        tick_value: float = 12.50,  # ES futures: $12.50 per tick (0.25 point)
        regime: str = 'unknown'
    ) -> Dict[str, float]:
        """
        Calculate position size based on confidence and risk parameters

        Args:
            confidence: Model confidence (0-1)
            account_balance: Current account balance ($)
            stop_loss_distance: Stop loss distance in points
            tick_value: Dollar value per tick
            regime: Market regime (for regime-based adjustments)

        Returns:
            Dictionary with:
                - contracts: Number of contracts to trade
                - risk_dollars: Dollar risk for this trade
                - risk_pct: Percentage of account at risk
                - confidence_scaled: Whether confidence scaling was applied
        """

        # Check minimum confidence
        if confidence < self.confidence_threshold:
            return {
                'contracts': 0,
                'risk_dollars': 0.0,
                'risk_pct': 0.0,
                'confidence_scaled': True,
                'reason': f'Confidence {confidence:.2%} below threshold {self.confidence_threshold:.2%}'
            }

        # Scale risk by confidence (linear scaling)
        # Confidence 0.65 -> base_risk_pct
        # Confidence 0.85+ -> max_risk_pct
        confidence_factor = (confidence - self.confidence_threshold) / (0.85 - self.confidence_threshold)
        confidence_factor = np.clip(confidence_factor, 0, 1)

        risk_pct = self.base_risk_pct + (self.max_risk_pct - self.base_risk_pct) * confidence_factor
        risk_pct = np.clip(risk_pct, self.min_risk_pct, self.max_risk_pct)

        # Regime-based adjustment (reduce size in choppy markets)
        regime_multipliers = {
            'trending_normal': 1.0,      # Full size
            'trending_high_vol': 0.8,    # Reduce 20%
            'ranging_normal': 0.7,       # Reduce 30%
            'ranging_low_vol': 0.5,      # Reduce 50%
            'high_vol_chaos': 0.4,       # Reduce 60%
            'transitional': 0.6,         # Reduce 40%
            'unknown': 0.7               # Conservative default
        }
        regime_multiplier = regime_multipliers.get(regime, 0.7)
        risk_pct *= regime_multiplier

        # Calculate dollar risk
        risk_dollars = account_balance * risk_pct

        # Calculate contracts needed
        # Risk per contract = stop_loss_distance * tick_value
        # (stop_loss_distance is in points, tick_value is $/tick)
        ticks_per_point = 4  # ES: 4 ticks per point (0.25 each)
        risk_per_contract = stop_loss_distance * ticks_per_point * tick_value

        if risk_per_contract <= 0:
            return {
                'contracts': 0,
                'risk_dollars': 0.0,
                'risk_pct': 0.0,
                'confidence_scaled': True,
                'reason': 'Invalid stop loss distance'
            }

        contracts = risk_dollars / risk_per_contract
        contracts = int(np.floor(contracts))  # Round down to whole contracts
        contracts = min(contracts, self.max_contracts)  # Apply maximum limit
        contracts = max(contracts, 1 if confidence >= self.confidence_threshold else 0)  # Minimum 1 if trading

        # Calculate actual risk with rounded contracts
        actual_risk_dollars = contracts * risk_per_contract
        actual_risk_pct = actual_risk_dollars / account_balance

        return {
            'contracts': contracts,
            'risk_dollars': actual_risk_dollars,
            'risk_pct': actual_risk_pct,
            'confidence_scaled': True,
            'regime_multiplier': regime_multiplier,
            'original_risk_pct': risk_pct / regime_multiplier  # Before regime adjustment
        }

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25  # Use 25% of Kelly (conservative)
    ) -> float:
        """
        Calculate Kelly Criterion position size

        Kelly Formula: f* = (p*b - q) / b
        where:
            p = win probability
            q = 1 - p
            b = avg_win / avg_loss

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade size
            avg_loss: Average losing trade size (positive number)
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)

        Returns:
            Recommended fraction of capital to risk (0-1)
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        b = avg_win / avg_loss  # Win/loss ratio
        p = win_rate
        q = 1 - p

        # Full Kelly
        kelly = (p * b - q) / b

        # Apply fractional Kelly (more conservative)
        fractional_kelly = kelly * kelly_fraction

        # Clamp to reasonable range
        return np.clip(fractional_kelly, 0.0, 0.05)  # Max 5% of capital


class StopTargetCalculator:
    """
    Calculate stop loss and take profit levels based on ATR and market regime
    """

    def __init__(
        self,
        base_stop_atr_multiplier: float = 0.75,  # TIGHTENED from 1.5
        base_target_atr_multiplier: float = 1.5   # TIGHTENED from 2.5
    ):
        self.base_stop_atr = base_stop_atr_multiplier
        self.base_target_atr = base_target_atr_multiplier

        # Regime-specific stop/target multipliers
        # TIGHTENED: Reduced all multipliers for tighter risk management
        # For ES futures with typical ATR of 10-15 points:
        # Stop: 0.5-1.0 ATR = 5-15 points = $62.50-$187.50
        # Target: 1.0-2.0 ATR = 10-30 points = $125-$375
        self.regime_params = {
            'trending_normal': {
                'stop_atr': 0.75,    # TIGHTENED from 1.5 (e.g., 7.5 points = $93.75)
                'target_atr': 2.0,   # TIGHTENED from 4.0 (e.g., 20 points = $250)
                'risk_reward': 2.67  # Maintains good R/R
            },
            'trending_high_vol': {
                'stop_atr': 1.0,     # TIGHTENED from 2.0 (wider for volatility)
                'target_atr': 2.5,   # TIGHTENED from 4.0
                'risk_reward': 2.5
            },
            'ranging_normal': {
                'stop_atr': 0.5,     # TIGHTENED from 1.0 (tight for ranging)
                'target_atr': 1.5,   # TIGHTENED from 2.5
                'risk_reward': 3.0   # Higher R/R compensates
            },
            'ranging_low_vol': {
                'stop_atr': 0.4,     # TIGHTENED from 0.8 (very tight for low vol)
                'target_atr': 1.2,   # TIGHTENED from 2.0
                'risk_reward': 3.0
            },
            'high_vol_chaos': {
                'stop_atr': 1.5,     # TIGHTENED from 2.5 (still wider for chaos)
                'target_atr': 3.0,   # TIGHTENED from 3.0 (kept same)
                'risk_reward': 2.0
            },
            'transitional': {
                'stop_atr': 0.6,     # TIGHTENED from 1.3
                'target_atr': 1.8,   # TIGHTENED from 3.0
                'risk_reward': 3.0
            },
            'unknown': {
                'stop_atr': 0.75,    # TIGHTENED from 1.5
                'target_atr': 2.0,   # TIGHTENED from 3.5
                'risk_reward': 2.67
            }
        }

    def calculate_stops(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        regime: str = 'unknown',
        confidence: float = 0.65,
        trend_direction: str = 'neutral',
        trend_strength: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate stop loss and take profit levels with trend-awareness

        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            atr: Average True Range (in points)
            regime: Market regime
            confidence: Model confidence (affects stop tightness)
            trend_direction: Direction of prevailing trend ('bullish', 'bearish', 'neutral')
            trend_strength: ADX value indicating trend strength

        Returns:
            Dictionary with:
                - stop_loss: Stop loss price
                - take_profit: Take profit price
                - stop_distance: Distance in points
                - target_distance: Distance in points
                - risk_reward: Ratio of target/stop
                - is_counter_trend: Whether this is a counter-trend trade
        """

        # Get regime parameters
        params = self.regime_params.get(regime, self.regime_params['unknown'])

        # Check if this is a counter-trend trade
        is_counter_trend = False
        if direction == 'long' and trend_direction == 'bearish':
            is_counter_trend = True
        elif direction == 'short' and trend_direction == 'bullish':
            is_counter_trend = True

        # Adjust risk/reward for counter-trend trades in strong trends
        target_multiplier = 1.0
        stop_multiplier = 1.0

        if is_counter_trend and trend_strength > 20:
            # REDUCED PENALTY: Targets reduced by max 20% (was 40%) for counter-trend trades
            # This allows counter-trend trades to capture more profit
            strength_factor = min(1.0, (trend_strength - 20) / 40)  # CHANGED: /40 instead of /20
            target_multiplier = 1.0 - (0.2 * strength_factor)  # CHANGED: 0.2 instead of 0.4

            print(f"⚠️  Counter-trend {direction} trade in {trend_direction} trend (ADX={trend_strength:.1f})")
            print(f"   Target reduced by {(1-target_multiplier)*100:.0f}% (multiplier={target_multiplier:.2f})")

        # Confidence-based adjustment (tighter stops for low confidence)
        # High confidence -> wider stops (let trade breathe)
        # Low confidence -> tighter stops (exit quickly if wrong)
        confidence_factor = 0.7 + (confidence - 0.65) * 1.5
        confidence_factor = np.clip(confidence_factor, 0.5, 1.2)

        # Calculate stop distance
        stop_distance = atr * params['stop_atr'] * confidence_factor * stop_multiplier

        # Calculate target distance with trend adjustment
        base_target = stop_distance * params['risk_reward']
        target_distance = base_target * target_multiplier

        # Round to nearest quarter point (ES tick size)
        stop_distance = round(stop_distance * 4) / 4
        target_distance = round(target_distance * 4) / 4

        # Calculate actual prices
        if direction == 'long':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + target_distance
        elif direction == 'short':
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - target_distance
        else:
            raise ValueError(f"Invalid direction: {direction}")

        # Round prices to quarter points
        stop_loss = round(stop_loss * 4) / 4
        take_profit = round(take_profit * 4) / 4

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_distance': stop_distance,
            'target_distance': target_distance,
            'risk_reward': target_distance / stop_distance if stop_distance > 0 else 0,
            'confidence_factor': confidence_factor,
            'is_counter_trend': is_counter_trend,
            'target_adjustment': target_multiplier,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'regime': regime
        }

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        direction: str,
        atr: float,
        initial_stop: float,
        trail_activation_rr: float = 1.0,  # Start trailing at 1R
        trail_distance_atr: float = 1.0    # Trail 1 ATR behind
    ) -> float:
        """
        Calculate trailing stop loss

        Args:
            entry_price: Original entry price
            current_price: Current market price
            direction: 'long' or 'short'
            atr: Current ATR
            initial_stop: Initial stop loss price
            trail_activation_rr: Start trailing after this many R (reward units)
            trail_distance_atr: Trail this many ATR behind price

        Returns:
            Updated stop loss price
        """
        initial_risk = abs(entry_price - initial_stop)

        if direction == 'long':
            profit = current_price - entry_price

            # Check if we've made enough profit to activate trailing
            if profit >= initial_risk * trail_activation_rr:
                # Trail ATR distance below current price
                new_stop = current_price - (atr * trail_distance_atr)
                # Never move stop down
                return max(new_stop, initial_stop)
            else:
                return initial_stop

        elif direction == 'short':
            profit = entry_price - current_price

            if profit >= initial_risk * trail_activation_rr:
                # Trail ATR distance above current price
                new_stop = current_price + (atr * trail_distance_atr)
                # Never move stop up
                return min(new_stop, initial_stop)
            else:
                return initial_stop

        return initial_stop


class CounterTrendFilter:
    """
    Filter or adjust counter-trend trades based on market regime

    This class helps prevent taking counter-trend trades in strong trending markets
    and favors counter-trend trades in ranging markets where they perform best.
    """

    def __init__(
        self,
        enable_filtering: bool = True,
        trending_adx_threshold: float = 25.0,
        counter_trend_confidence_penalty: float = 0.5,  # Reduce confidence by 50% in trends
        ranging_confidence_boost: float = 0.15,  # Boost confidence by 15% in ranging
        block_counter_trends_in_strong_trends: bool = False  # CHANGED: False to allow counter-trend trades
    ):
        self.enable_filtering = enable_filtering
        self.trending_adx_threshold = trending_adx_threshold
        self.counter_trend_confidence_penalty = counter_trend_confidence_penalty
        self.ranging_confidence_boost = ranging_confidence_boost
        self.block_counter_trends = block_counter_trends_in_strong_trends

    def is_counter_trend(self, signal: str, trend_direction: str) -> bool:
        """Check if signal is counter to the prevailing trend"""
        if trend_direction == 'neutral':
            return False

        if signal == 'long' and trend_direction == 'bearish':
            return True
        if signal == 'short' and trend_direction == 'bullish':
            return True

        return False

    def filter_signal(
        self,
        signal: str,
        confidence: float,
        regime_info: dict
    ) -> tuple:
        """
        Filter or adjust counter-trend signals based on regime

        Args:
            signal: Trading signal ('long', 'short', 'hold')
            confidence: Model confidence (0-1)
            regime_info: Dict with 'regime', 'trend_direction', 'trend_strength'

        Returns:
            (adjusted_signal, adjusted_confidence, filter_details)
        """
        if not self.enable_filtering or signal == 'hold':
            return signal, confidence, {'filtered': False, 'reason': 'No filtering needed'}

        regime = regime_info.get('regime', 'unknown')
        trend_direction = regime_info.get('trend_direction', 'neutral')
        trend_strength = regime_info.get('trend_strength', 0.0)

        is_counter = self.is_counter_trend(signal, trend_direction)

        # CASE 1: Strong Trending Market + Counter-Trend Signal
        if is_counter and trend_strength > self.trending_adx_threshold:
            if regime in ['trending_normal', 'trending_high_vol']:
                if self.block_counter_trends:
                    # Block counter-trend trades entirely in strong trends
                    return 'hold', 0.0, {
                        'filtered': True,
                        'reason': f'Counter-trend {signal} BLOCKED in {regime} (ADX={trend_strength:.1f})',
                        'original_signal': signal,
                        'original_confidence': confidence,
                        'trend_direction': trend_direction
                    }
                else:
                    # Reduce confidence significantly but allow the trade
                    adjusted_confidence = confidence * (1 - self.counter_trend_confidence_penalty)
                    return signal, adjusted_confidence, {
                        'filtered': True,
                        'reason': f'Counter-trend confidence reduced in {regime}',
                        'confidence_penalty': self.counter_trend_confidence_penalty,
                        'trend_direction': trend_direction
                    }

        # CASE 2: Ranging Market + Counter-Trend Signal = GOOD
        if is_counter and trend_strength < 20:
            if regime in ['ranging_normal', 'ranging_low_vol']:
                # Boost confidence for counter-trend trades in ranging markets
                confidence_boost = min(self.ranging_confidence_boost, (20 - trend_strength) / 100)
                adjusted_confidence = min(1.0, confidence + confidence_boost)

                return signal, adjusted_confidence, {
                    'filtered': False,
                    'boosted': True,
                    'reason': f'Counter-trend {signal} FAVORED in {regime}',
                    'confidence_boost': confidence_boost,
                    'trend_direction': trend_direction
                }

        # CASE 3: Transitional markets - reduce all confidence slightly
        if regime == 'transitional':
            adjusted_confidence = confidence * 0.85
            return signal, adjusted_confidence, {
                'filtered': True,
                'reason': 'Confidence reduced in transitional regime',
                'confidence_penalty': 0.15,
                'trend_direction': trend_direction
            }

        # CASE 4: High volatility chaos - reduce confidence
        if regime == 'high_vol_chaos':
            adjusted_confidence = confidence * 0.7
            return signal, adjusted_confidence, {
                'filtered': True,
                'reason': 'Confidence reduced in high volatility chaos',
                'confidence_penalty': 0.3,
                'trend_direction': trend_direction
            }

        # Default: no change
        return signal, confidence, {
            'filtered': False,
            'reason': 'Signal passed filtering',
            'is_counter_trend': is_counter,
            'trend_direction': trend_direction
        }


class RiskManager:
    """
    Combined risk management: position sizing + stop/target calculation
    """

    def __init__(
        self,
        position_sizer: Optional[PositionSizer] = None,
        stop_calculator: Optional[StopTargetCalculator] = None,
        counter_trend_filter: Optional[CounterTrendFilter] = None
    ):
        self.position_sizer = position_sizer or PositionSizer()
        self.stop_calculator = stop_calculator or StopTargetCalculator()
        self.counter_trend_filter = counter_trend_filter or CounterTrendFilter()

    def calculate_trade_parameters(
        self,
        signal: str,
        confidence: float,
        entry_price: float,
        atr: float,
        regime: str,
        account_balance: float,
        tick_value: float = 12.50,
        regime_info: Optional[dict] = None
    ) -> Dict:
        """
        Calculate complete trade parameters: position size, stops, targets
        Now includes counter-trend filtering and trend-aware stop/target adjustment

        Args:
            signal: 'long', 'short', or 'hold'
            confidence: Model confidence (0-1)
            entry_price: Proposed entry price
            atr: Average True Range
            regime: Market regime (string for backward compatibility)
            account_balance: Account balance
            tick_value: Value per tick
            regime_info: Enhanced regime info dict with trend_direction, trend_strength (optional)

        Returns:
            Complete trade specification with all parameters
        """

        # Extract trend info from regime_info or use defaults
        if regime_info:
            trend_direction = regime_info.get('trend_direction', 'neutral')
            trend_strength = regime_info.get('trend_strength', 0.0)
        else:
            trend_direction = 'neutral'
            trend_strength = 0.0
            regime_info = {
                'regime': regime,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength
            }

        if signal == 'hold':
            return {
                'signal': 'hold',
                'confidence': confidence,
                'contracts': 0,
                'entry_price': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'reason': 'Hold signal',
                'regime': regime,
                'trend_direction': trend_direction
            }

        # Apply counter-trend filter
        filtered_signal, filtered_confidence, filter_details = self.counter_trend_filter.filter_signal(
            signal, confidence, regime_info
        )

        # If signal was blocked, return hold
        if filtered_signal == 'hold':
            return {
                'signal': 'hold',
                'confidence': filtered_confidence,
                'contracts': 0,
                'entry_price': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'reason': filter_details.get('reason', 'Filtered by counter-trend filter'),
                'regime': regime,
                'trend_direction': trend_direction,
                'filter_details': filter_details,
                'original_signal': signal,
                'original_confidence': confidence
            }

        # Calculate stops with trend-awareness
        stops = self.stop_calculator.calculate_stops(
            entry_price, filtered_signal, atr, regime, filtered_confidence,
            trend_direction=trend_direction,
            trend_strength=trend_strength
        )

        # Calculate position size
        position = self.position_sizer.calculate_position_size(
            filtered_confidence,
            account_balance,
            stops['stop_distance'],
            tick_value,
            regime
        )

        # Combine results
        return {
            'signal': filtered_signal,
            'confidence': filtered_confidence,
            'original_confidence': confidence,
            'contracts': position['contracts'],
            'entry_price': entry_price,
            'stop_loss': stops['stop_loss'],
            'take_profit': stops['take_profit'],
            'stop_distance': stops['stop_distance'],
            'target_distance': stops['target_distance'],
            'risk_reward': stops['risk_reward'],
            'risk_dollars': position['risk_dollars'],
            'risk_pct': position['risk_pct'],
            'regime': regime,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'is_counter_trend': stops.get('is_counter_trend', False),
            'target_adjustment': stops.get('target_adjustment', 1.0),
            'regime_multiplier': position.get('regime_multiplier', 1.0),
            'confidence_factor': stops['confidence_factor'],
            'filter_details': filter_details,
            'reason': position.get('reason', 'Trade approved')
        }


# Example usage
if __name__ == '__main__':
    # Initialize risk manager
    risk_mgr = RiskManager()

    # Example trade parameters
    trade_params = risk_mgr.calculate_trade_parameters(
        signal='long',
        confidence=0.72,
        entry_price=4500.00,
        atr=15.0,  # 15 points ATR
        regime='trending_normal',
        account_balance=25000.00
    )

    print("Trade Parameters:")
    print(f"  Signal: {trade_params['signal'].upper()}")
    print(f"  Contracts: {trade_params['contracts']}")
    print(f"  Entry: ${trade_params['entry_price']:.2f}")
    print(f"  Stop Loss: ${trade_params['stop_loss']:.2f}")
    print(f"  Take Profit: ${trade_params['take_profit']:.2f}")
    print(f"  Risk/Reward: {trade_params['risk_reward']:.2f}")
    print(f"  Risk: ${trade_params['risk_dollars']:.2f} ({trade_params['risk_pct']:.2%})")
    print(f"  Regime: {trade_params['regime']}")
