# Git Hooks

This directory contains custom git hooks for the backtester-pro project.

## Available Hooks

### pre-commit

**Purpose**: Validates Python syntax before allowing commits

**What it does**:
- Checks all staged Python files for syntax errors
- Uses `python3 -m py_compile` to validate
- Blocks commit if any errors are found
- Cleans up `.pyc` files automatically

## Installation

### Option 1: Configure Git to Use This Directory

```bash
cd /home/user/backtester-pro
git config core.hooksPath .githooks
```

This tells git to use hooks from `.githooks/` instead of `.git/hooks/`

### Option 2: Copy Hooks Manually

```bash
cd /home/user/backtester-pro
cp .githooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Verify Installation

```bash
# Check if hooks path is configured
git config core.hooksPath

# Or check if hook is executable
ls -la .git/hooks/pre-commit
```

## Usage

Once installed, the hooks run automatically:

```bash
# Make changes to Python files
vim rnn-server/model.py

# Stage changes
git add rnn-server/model.py

# Commit (hook runs automatically)
git commit -m "Update model"

# If syntax errors exist:
#   ✗ COMMIT REJECTED: Python syntax errors detected
#
# If all files are valid:
#   ✓ All Python files validated successfully
#   [commit succeeds]
```

## Bypassing Hooks (Not Recommended)

If you need to bypass the hook (for emergency commits):

```bash
git commit --no-verify -m "Emergency fix"
```

**Warning**: Only use `--no-verify` when absolutely necessary!

## Testing Hooks

Test the pre-commit hook without making a commit:

```bash
# Stage a Python file
git add some_file.py

# Run hook manually
.githooks/pre-commit

# Unstage if needed
git reset HEAD some_file.py
```

## Adding New Hooks

To add a new hook:

1. Create hook file in `.githooks/` (e.g., `pre-push`)
2. Make it executable: `chmod +x .githooks/pre-push`
3. Add documentation here
4. Test it thoroughly

## Available Git Hooks

Git supports many hook types:
- `pre-commit` - Before commit is created
- `prepare-commit-msg` - Before commit message editor opens
- `commit-msg` - After commit message is entered
- `post-commit` - After commit is created
- `pre-push` - Before push to remote
- `pre-rebase` - Before rebase
- And more...

See: https://git-scm.com/docs/githooks

## Troubleshooting

### Hook not running

```bash
# Check if hooks path is configured
git config core.hooksPath

# Should output: .githooks
# If empty, run:
git config core.hooksPath .githooks
```

### Permission denied

```bash
# Make hook executable
chmod +x .githooks/pre-commit
```

### Hook fails incorrectly

```bash
# Test hook manually
.githooks/pre-commit

# Check Python is available
which python3
python3 --version
```

## Benefits

✅ **Catch errors early** - Before they reach the repository
✅ **Consistent quality** - Automated checks for all commits
✅ **Save time** - No broken commits in history
✅ **Team standards** - Enforce project conventions

## Best Practices

1. **Keep hooks fast** - Should complete in <5 seconds
2. **Clear output** - Show what's being checked
3. **Helpful errors** - Explain how to fix issues
4. **Allow bypass** - For emergencies (with `--no-verify`)
5. **Document well** - Explain what each hook does
