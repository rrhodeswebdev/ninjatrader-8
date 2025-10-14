# Windows Executable Build Guide

This guide shows you how to automatically build a Windows executable for your RNN Trading Server using GitHub Actions.

## What You Get

✅ **Automatic Windows builds** - Every time you push to GitHub
✅ **No manual building** - GitHub does it for you in the cloud
✅ **~60 minute builds** - Builds happen in the background
✅ **Easy distribution** - Download a single zip file with everything

## Quick Start

### 1. Push to GitHub

```bash
cd /Users/ryanrhodes/projects/ninjatrader-8

# Add and commit everything
git add .
git commit -m "Add Windows executable build workflow"

# Push to GitHub (replace with your repo URL)
git push origin main
```

### 2. Watch the Build

1. Go to your GitHub repository
2. Click the **Actions** tab
3. You'll see "Build RNN Server Executables" running
4. Wait ~60 minutes for completion

### 3. Download the Executable

**Option A - From Actions (for testing):**
1. Actions tab → Click on completed workflow run
2. Scroll to **Artifacts** section
3. Download `rnn-server-windows.zip`

**Option B - Create a Release (for distribution):**
```bash
# Create and push a version tag
git tag v1.0.0
git push origin v1.0.0
```

Then:
1. Go to **Releases** tab on GitHub
2. Click the latest release
3. Download `rnn-server-windows.zip` from Assets

## What's in the Zip File

```
rnn-server-windows.zip
└── server_app.dist/
    ├── rnn-server.exe          ← Your executable!
    ├── models/                 ← Trained models
    ├── torch/                  ← PyTorch libraries
    ├── pandas/                 ← Data processing
    └── [all dependencies]      ← Everything needed
```

## Running the Executable

### On Windows:

1. Extract the zip file
2. Open the `server_app.dist` folder
3. **Double-click** `rnn-server.exe`

OR from Command Prompt:

```cmd
cd path\to\server_app.dist
rnn-server.exe
```

The server starts on `http://127.0.0.1:8000`

### Configuration

Set environment variables before running:

```cmd
set PORT=8080
set HOST=0.0.0.0
rnn-server.exe
```

## Build Times & Costs

### Build Time
- **First build:** ~60 minutes
- **Subsequent builds:** ~50-60 minutes
- **Builds run in the cloud** - You don't have to wait!

### GitHub Actions Costs
- **Public repositories:** FREE ✅
- **Private repositories:** 2,000 minutes/month FREE
  - Each build uses ~60 minutes
  - Can do ~33 builds/month for free

## How to Trigger Builds

### Automatic Builds

Builds start automatically when you:

```bash
# Push to main branch
git push origin main
```

### Release Builds

Create a permanent release:

```bash
# Tag a version
git tag v1.0.0
git push origin v1.0.0
```

This creates a GitHub Release with the executable attached.

### Manual Builds

1. Go to Actions tab
2. Click "Build RNN Server Executables"
3. Click "Run workflow"
4. Select "main" branch
5. Click "Run workflow" button

## Distribution to Users

### For End Users:

1. **Share the Release URL:**
   ```
   https://github.com/YOUR_USERNAME/ninjatrader-8/releases
   ```

2. **Users download** `rnn-server-windows.zip`

3. **Users extract and run** - No Python or setup needed!

### What Users Need:
- ✅ Windows 10 or later
- ✅ Nothing else! (All dependencies included)

## Updating the Executable

When you make code changes:

```bash
# 1. Make your changes
# ... edit code ...

# 2. Commit and push
git add .
git commit -m "Update model or fix bug"
git push origin main

# 3. Wait for build (~60 minutes)

# 4. Download new executable from Actions

# 5. Ready for a release?
git tag v1.0.1
git push origin v1.0.1
```

## Troubleshooting

### Build Failed?

1. Go to Actions tab → Click failed workflow
2. Click on "Build Windows Executable" job
3. Read the error logs
4. Common issues:
   - Syntax error in Python code
   - Missing dependency in `pyproject.toml`
   - Build script error

### Can't Download Executable?

- Make sure build completed (green checkmark)
- Artifacts expire after 30 days
- Use Releases for permanent downloads

### Executable Won't Run on Windows?

- User needs Windows 10 or later
- Windows Defender might block it (click "More info" → "Run anyway")
- Firewall might ask for permission (click "Allow")

## File Reference

Your project now has these files:

- `.github/workflows/build-executables.yml` - GitHub Actions workflow
- `rnn-server/build_executable.bat` - Windows build script
- `rnn-server/run_server.bat` - Quick run script
- `rnn-server/server_app.py` - Executable entry point

## Example Workflow

```
Monday 9am:   Make code changes
Monday 10am:  git push origin main
              → Build starts automatically

Monday 11am:  Build completes
              → Download from Actions tab
              → Test the executable

Tuesday 2pm:  Everything works!
              git tag v1.0.0
              git push origin v1.0.0
              → Release created

Tuesday 3pm:  Share release URL with users
              → Users download and run!
```

## Benefits

✅ **No Windows machine needed** - Build in the cloud
✅ **Consistent builds** - Same environment every time
✅ **Easy updates** - Just push code, get new executable
✅ **Professional** - GitHub Releases look polished
✅ **Version history** - Track all releases

## Next Steps

1. ✅ Push code to GitHub
2. ⏳ Wait for first build (check Actions tab)
3. ⏳ Download and test executable
4. ⏳ Create release when ready
5. ✅ Share with users!

## Support

- **Actions logs:** GitHub repo → Actions tab → Click workflow run
- **Workflow file:** `.github/workflows/build-executables.yml`
- **Build script:** `rnn-server/build_executable.bat`

---

**Ready?** Just push to GitHub and the build will start automatically! 🚀
