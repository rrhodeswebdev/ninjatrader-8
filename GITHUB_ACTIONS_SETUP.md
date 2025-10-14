# GitHub Actions Setup Guide

This guide shows you how to set up automated builds for Windows and macOS executables.

## Quick Start

### Step 1: Push to GitHub

If you haven't already, push your code to GitHub:

```bash
cd /Users/ryanrhodes/projects/ninjatrader-8

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit with GitHub Actions workflow"

# Add your GitHub remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/ninjatrader-8.git

# Push to GitHub
git push -u origin main
```

### Step 2: Enable GitHub Actions

1. Go to your repository on GitHub
2. Click **Actions** tab
3. If prompted, click **"I understand my workflows, go ahead and enable them"**

That's it! GitHub Actions is now enabled.

### Step 3: Trigger Your First Build

You have three options:

#### Option A: Push to Main (Automatic)

```bash
# Make any change
echo "# RNN Server" > rnn-server/README.md
git add .
git commit -m "Trigger build"
git push origin main
```

The build will start automatically!

#### Option B: Create a Release Tag

```bash
# Create and push a tag
git tag v1.0.0
git push origin v1.0.0
```

This will build executables AND create a GitHub Release!

#### Option C: Manual Trigger

1. Go to GitHub → Your Repo → **Actions** tab
2. Click **Build RNN Server Executables** (left sidebar)
3. Click **Run workflow** (right side)
4. Select branch: `main`
5. Click **Run workflow** button

## What Happens Next

### Build Process (60 minutes)

1. **GitHub Actions starts two jobs** (parallel):
   - Windows build on `windows-latest`
   - macOS build on `macos-latest`

2. **Each job:**
   - Sets up Python 3.11
   - Installs `uv` package manager
   - Installs dependencies (PyTorch, FastAPI, etc.)
   - Runs Nuitka to build executable
   - Creates zip file
   - Uploads artifact

3. **After ~60 minutes:**
   - Both executables are ready
   - Available in Actions → Artifacts section

### Download Your Executables

#### From Actions Tab:

1. Go to **Actions** tab
2. Click on the latest successful workflow run (green checkmark)
3. Scroll down to **Artifacts** section
4. Download:
   - `rnn-server-windows.zip`
   - `rnn-server-macos.zip`

#### From Releases (if you used a tag):

1. Go to **Releases** tab
2. Click on the latest release
3. Download from **Assets** section:
   - `rnn-server-windows.zip`
   - `rnn-server-macos.zip`

## Monitoring the Build

### Watch Build Progress:

1. Go to **Actions** tab
2. Click on the running workflow
3. Click on **Windows** or **macOS** job
4. Watch live logs as the build progresses

### Build Status Indicators:

- 🟡 Yellow circle = Running
- ✅ Green checkmark = Success
- ❌ Red X = Failed

### Estimated Times:

- **Setup (Python, uv, dependencies):** ~5 minutes
- **Nuitka compilation:** ~50-55 minutes
- **Total per platform:** ~60 minutes
- **Both platforms (parallel):** ~60 minutes total

## Using the Built Executables

### Windows:

1. Download `rnn-server-windows.zip`
2. Extract the zip file
3. Navigate to `server_app.dist\`
4. Double-click `rnn-server.exe` or run from cmd:
   ```cmd
   rnn-server.exe
   ```

### macOS:

1. Download `rnn-server-macos.zip`
2. Extract the zip file
3. Open Terminal and navigate to the folder:
   ```bash
   cd path/to/server_app.dist
   chmod +x rnn-server  # Make executable (first time only)
   ./rnn-server
   ```

## Creating Releases

### Automatic Release Creation:

```bash
# Create an annotated tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push the tag
git push origin v1.0.0
```

GitHub Actions will:
1. Build Windows and macOS executables
2. Create a new GitHub Release
3. Attach both zip files
4. Add release notes automatically

### Versioning Scheme:

- `v1.0.0` - Major release
- `v1.1.0` - Minor release (new features)
- `v1.0.1` - Patch release (bug fixes)
- `v1.0.0-beta` - Pre-release

## Troubleshooting

### Build Failed?

1. **Check the logs:**
   - Actions → Failed workflow → Click on the job → View logs

2. **Common issues:**
   - **Python package conflicts:** Check `pyproject.toml` dependencies
   - **Build script errors:** Test locally first
   - **Out of memory:** Normal for large projects, should work on GitHub runners
   - **Timeout:** Shouldn't happen (6 hour limit), but check for infinite loops

3. **Fix and retry:**
   ```bash
   # Fix the issue locally
   # Test the build script
   cd rnn-server
   ./build_executable.sh  # or build_executable.bat on Windows

   # Push the fix
   git add .
   git commit -m "Fix build issue"
   git push origin main
   ```

### Artifacts Not Showing?

- Wait for build to complete (green checkmark)
- Artifacts appear only after successful build
- Artifacts expire after 30 days

### Release Not Created?

- Make sure you pushed a **tag** starting with `v`
- Check that tag was pushed: `git push origin v1.0.0`
- Verify tag exists on GitHub: Repo → Tags

## Advanced Configuration

### Modify Build Settings:

Edit `.github/workflows/build-executables.yml`:

```yaml
# Change Python version
python-version: '3.11'  # or '3.10', '3.12'

# Add Linux builds
matrix:
  os: [windows-latest, macos-latest, ubuntu-latest]

# Change retention period
retention-days: 90  # Keep artifacts for 90 days
```

### Customize Release Notes:

Edit the `body:` section in the workflow file to change the release description.

### Add Build Notifications:

Use GitHub's notification settings or add a Slack/Discord webhook.

## Cost and Limits

### Free Tier (Public Repos):
- ✅ Unlimited minutes
- ✅ Unlimited storage (for 30 days)
- ✅ Unlimited concurrent jobs

### Free Tier (Private Repos):
- ✅ 2,000 minutes/month
- ✅ 500 MB storage
- Each build uses ~120 minutes (60 min × 2 platforms)
- Can do ~16 builds per month

### Paid Plans:
- If you need more, GitHub Pro/Team/Enterprise have higher limits

## Testing Locally Before Pushing

Always test your build scripts locally:

**macOS:**
```bash
cd rnn-server
./build_executable.sh
```

**Windows:**
```cmd
cd rnn-server
build_executable.bat
```

This ensures the GitHub Actions workflow will succeed!

## Next Steps

1. ✅ Push code to GitHub
2. ✅ Wait for first build (~60 minutes)
3. ✅ Download executables from Artifacts
4. ✅ Test the executables on target platforms
5. ✅ Create a release tag when ready
6. ✅ Share the release URL with users!

## Example Workflow

```bash
# 1. Develop locally
# ... make changes ...

# 2. Test build locally
cd rnn-server
./build_executable.sh

# 3. Commit and push
git add .
git commit -m "Add new feature"
git push origin main

# 4. Wait for GitHub Actions to build

# 5. When ready for release:
git tag v1.0.0
git push origin v1.0.0

# 6. Share the release!
# GitHub automatically creates release at:
# https://github.com/YOUR_USERNAME/ninjatrader-8/releases/tag/v1.0.0
```

## Support

- GitHub Actions Documentation: https://docs.github.com/en/actions
- Nuitka Documentation: https://nuitka.net/
- Workflow file: `.github/workflows/build-executables.yml`
- Workflow README: `.github/workflows/README.md`

Happy building! 🚀
