#!/bin/bash

echo "=========================================="
echo "Verifying Trading Model Optimizations"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

# Test function
test_change() {
    local description="$1"
    local command="$2"
    local expected="$3"
    
    echo -n "Testing: $description... "
    
    result=$(eval "$command")
    
    if echo "$result" | grep -q "$expected"; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo "  Expected: $expected"
        echo "  Got: $result"
        ((FAILED++))
    fi
}

echo "1. Checking Early Exit Thresholds (model.py)"
echo "--------------------------------------------"
test_change "Opposite signal threshold (0.65)" "grep -A 1 'current_position.*long.*signal.*short.*confidence' model.py | head -1" "0.65"
test_change "HOLD probability threshold (0.70)" "grep 'prob_hold > 0' model.py | grep -v '#'" "0.70"
test_change "Directional reversal margin (0.15)" "grep 'prob_short > prob_long +' model.py | head -1" "0.15"
test_change "Momentum loss bars (5)" "grep 'len(recent_bars_df) >= [0-9]' model.py | tail -1" ">= 5"
echo ""

echo "2. Checking Risk/Reward Targets (risk_management.py)"
echo "-----------------------------------------------------"
test_change "Trending normal target (4.0)" "grep -A 2 \"'trending_normal'\" risk_management.py | grep target_atr" "4.0"
test_change "Ranging normal target (2.5)" "grep -A 2 \"'ranging_normal'\" risk_management.py | grep target_atr" "2.5"
test_change "Counter-trend penalty (0.2)" "grep 'target_multiplier = 1.0 - ' risk_management.py | grep -v '#'" "0.2"
echo ""

echo "3. Checking Counter-Trend Filter (risk_management.py)"
echo "------------------------------------------------------"
test_change "Counter-trend blocking disabled" "grep 'block_counter_trends_in_strong_trends.*bool.*=' risk_management.py | head -1" "False"
echo ""

echo "4. Checking Trailing Stop Implementation (main.py)"
echo "---------------------------------------------------"
test_change "Trailing stop calculation" "grep 'calculate_trailing_stop' main.py" "calculate_trailing_stop"
test_change "Trailing stop in response" "grep 'trailing_stop' main.py | grep -v '#'" "trailing_stop"
echo ""

echo "5. Checking Training Parameters (train_phase3.py)"
echo "--------------------------------------------------"
test_change "Default epochs (150)" "grep 'epochs: int = ' train_phase3.py | head -1" "150"
test_change "Argument default epochs (150)" "grep \"default=.*help='Training epochs\" train_phase3.py" "150"
echo ""

echo "6. Checking Backup Files"
echo "------------------------"
if [ -f "models/trading_model_backup_20251021_160130.pth" ]; then
    echo -e "${GREEN}✓ Backup model exists${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ Backup model NOT found${NC}"
    ((FAILED++))
fi
echo ""

echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo -e "Tests Passed: ${GREEN}${PASSED}${NC}"
echo -e "Tests Failed: ${RED}${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All optimizations verified successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Start paper trading to test changes"
    echo "2. Monitor: Average hold time, exit reasons, profit per trade"
    echo "3. After 50 trades, evaluate performance"
    echo "4. If successful, retrain with: uv run python train_phase3.py --all --rth-only --epochs 150"
    exit 0
else
    echo -e "${YELLOW}⚠ Some checks failed. Review the output above.${NC}"
    exit 1
fi
