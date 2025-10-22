#!/bin/bash

# RNN Server Legacy File Cleanup Script
# Safely moves unused/duplicate files to backup folder

set -e  # Exit on error

BACKUP_DIR="_legacy_backup_$(date +%Y%m%d_%H%M%S)"

echo "🧹 RNN Server Cleanup Script"
echo "=============================="
echo ""
echo "This script will move legacy/unused files to: $BACKUP_DIR"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cleanup cancelled."
    exit 1
fi

# Create backup directory
echo "📦 Creating backup directory..."
mkdir -p "$BACKUP_DIR"

# Counter for moved files
MOVED=0

# Function to move file with logging
move_file() {
    if [ -f "$1" ]; then
        echo "  Moving: $1"
        mv "$1" "$BACKUP_DIR/"
        ((MOVED++))
    else
        echo "  ⚠️  Not found: $1 (skipping)"
    fi
}

echo ""
echo "🗂️  Moving legacy training scripts..."
move_file "train_phase3.py"
move_file "train_ensemble.py"
move_file "deploy_improvements.py"
move_file "deploy_round2_improvements.py"

echo ""
echo "🗂️  Moving old optimization files..."
move_file "feature_optimization.py"
move_file "feature_importance.py"
move_file "feature_importance_analyzer.py"
move_file "hyperparameter_optimization.py"
move_file "performance_optimization.py"

echo ""
echo "🗂️  Moving old model variations..."
move_file "model_simplified.py"
move_file "model_enhancements.py"

echo ""
echo "🗂️  Moving unused ensemble/advanced features..."
move_file "ensemble.py"
move_file "ensemble_advanced.py"
move_file "meta_labeling.py"
move_file "regime_models.py"

echo ""
echo "🗂️  Moving old augmentation/learning files..."
move_file "advanced_augmentation.py"
move_file "advanced_loss_functions.py"
move_file "curriculum_learning.py"

echo ""
echo "🗂️  Moving old calibration files..."
move_file "confidence_calibration.py"
move_file "confidence_calibration_advanced.py"
move_file "calibrate_model.py"

echo ""
echo "🗂️  Moving old validation files..."
move_file "walk_forward_optimizer.py"

echo ""
echo "🗂️  Moving old risk management..."
move_file "risk_management.py"

echo ""
echo "🗂️  Moving unused monitoring/retraining..."
move_file "monitoring_dashboard.py"
move_file "adaptive_retraining.py"

echo ""
echo "🗂️  Moving duplicate server file..."
move_file "server_app.py"

echo ""
echo "=============================="
echo "✅ Cleanup Complete!"
echo "=============================="
echo ""
echo "📊 Summary:"
echo "  Files moved: $MOVED"
echo "  Backup location: $BACKUP_DIR/"
echo ""
echo "🔍 Remaining core files:"
echo ""
ls -lh *.py 2>/dev/null | grep -E "(main|simplified_model|orderflow|price_action|regime|probability|adaptive_risk|performance|walk_forward|data_augmentation|train_improved)" || echo "  (Run 'ls *.py' to see all files)"
echo ""
echo "📝 Notes:"
echo "  - All files safely backed up (not deleted)"
echo "  - To restore: mv $BACKUP_DIR/* ."
echo "  - To permanently delete backup: rm -rf $BACKUP_DIR"
echo ""
echo "🎯 Next steps:"
echo "  1. Test your /analysis endpoint: uv run fastapi dev main.py"
echo "  2. If everything works, you can delete $BACKUP_DIR"
echo "  3. If issues arise, restore from backup"
echo ""
