# Release Workflow - Visual Guide

## How GitHub Actions Builds Your Executables

```
┌─────────────────────────────────────────────────────────────┐
│  YOU: Push code or create tag                               │
│                                                             │
│  $ git push origin main                                     │
│  OR                                                         │
│  $ git tag v1.0.0 && git push origin v1.0.0                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  GITHUB: Detects push/tag and triggers workflow             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├──────────────────┬──────────────────┐
                     ▼                  ▼                  ▼
         ┌──────────────────┐ ┌──────────────────┐       │
         │  Windows Build   │ │   macOS Build    │       │
         │                  │ │                  │       │
         │  windows-latest  │ │  macos-latest    │       │
         └────────┬─────────┘ └────────┬─────────┘       │
                  │                    │                  │
                  │ (parallel)         │                  │
                  ▼                    ▼                  ▼
         ┌──────────────────┐ ┌──────────────────┐
         │ 1. Setup Python  │ │ 1. Setup Python  │
         │ 2. Install uv    │ │ 2. Install uv    │
         │ 3. Install deps  │ │ 3. Install deps  │
         │ 4. Run Nuitka    │ │ 4. Run Nuitka    │
         │ 5. Create zip    │ │ 5. Create zip    │
         │ (~60 minutes)    │ │ (~60 minutes)    │
         └────────┬─────────┘ └────────┬─────────┘
                  │                    │
                  ▼                    ▼
         ┌──────────────────┐ ┌──────────────────┐
         │ rnn-server.exe   │ │   rnn-server     │
         │ (Windows)        │ │   (macOS)        │
         └────────┬─────────┘ └────────┬─────────┘
                  │                    │
                  └──────────┬─────────┘
                             ▼
                  ┌──────────────────────┐
                  │  Upload Artifacts    │
                  │                      │
                  │  - Windows zip       │
                  │  - macOS zip         │
                  └──────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌──────────────────┐         ┌──────────────────┐
    │  If Push to Main │         │  If Version Tag  │
    │                  │         │                  │
    │  → Artifacts     │         │  → Create        │
    │    available in  │         │    GitHub        │
    │    Actions tab   │         │    Release       │
    │                  │         │                  │
    │  (30 days)       │         │  (permanent)     │
    └──────────────────┘         └──────────────────┘
```

## Release Download Flow

```
┌─────────────────────────────────────────────────────────┐
│  USERS: Want to download the RNN Server                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ Go to GitHub Releases │
         │                       │
         │ github.com/you/repo/  │
         │ releases              │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ Click Latest Release  │
         │                       │
         │ v1.0.0                │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────────────────┐
         │ Download Assets:                  │
         │                                   │
         │ 📦 rnn-server-windows.zip (500MB) │
         │ 📦 rnn-server-macos.zip (500MB)   │
         └───────────┬───────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ Extract zip file      │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────────────┐
         │ Run the executable:           │
         │                               │
         │ Windows: rnn-server.exe       │
         │ macOS: ./rnn-server           │
         └───────────┬───────────────────┘
                     │
                     ▼
         ┌───────────────────────────────┐
         │ Server starts! 🚀             │
         │                               │
         │ http://127.0.0.1:8000         │
         └───────────────────────────────┘
```

## Your Development Workflow

```
┌─────────────────────────────────────────────────────────┐
│  LOCAL DEVELOPMENT                                      │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ 1. Write code         │
         │    (main.py, etc.)    │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ 2. Test locally       │
         │    uv run fastapi dev │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ 3. Optional:          │
         │    Test build locally │
         │    ./build_exec*.sh   │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ 4. Commit & push      │
         │    git push origin    │
         │    main               │
         └───────────┬───────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  GITHUB ACTIONS (Automatic)                             │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ 5. Builds executables │
         │    (~60 minutes)      │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ 6. Artifacts ready    │
         │    Download & test    │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ 7. Ready to release?  │
         │    Create tag:        │
         │    git tag v1.0.0     │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ 8. GitHub Release     │
         │    created!           │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ 9. Users download     │
         │    & use!             │
         └───────────────────────┘
```

## Build Status Badges

Add these to your README.md to show build status:

```markdown
![Build Status](https://github.com/YOUR_USERNAME/ninjatrader-8/actions/workflows/build-executables.yml/badge.svg)

[![Windows Build](https://github.com/YOUR_USERNAME/ninjatrader-8/actions/workflows/build-executables.yml/badge.svg?branch=main&event=push)](https://github.com/YOUR_USERNAME/ninjatrader-8/actions)
```

## Quick Reference Commands

### Trigger Regular Build
```bash
git push origin main
```
→ Builds executables, uploads to Actions artifacts (30 days)

### Create Release
```bash
git tag v1.0.0
git push origin v1.0.0
```
→ Builds executables, creates GitHub Release (permanent)

### Manual Build
1. Go to Actions tab
2. Click "Build RNN Server Executables"
3. Click "Run workflow"

### Download Artifacts
1. Actions → Workflow run → Artifacts section
2. Download zip files

### Download Release
1. Releases tab → Latest release
2. Download from Assets

## Timeline Example

```
Monday 9:00 AM    │ You: git push origin main
                  │
Monday 9:01 AM    │ GitHub: Build started
                  │
                  │ ⏳ Windows building...
                  │ ⏳ macOS building...
                  │
Monday 10:00 AM   │ ✅ Both builds complete
                  │
Monday 10:05 AM   │ You: Download artifacts from Actions
                  │ You: Test on both platforms
                  │
Monday 2:00 PM    │ You: Everything works!
                  │ You: git tag v1.0.0
                  │ You: git push origin v1.0.0
                  │
Monday 2:01 PM    │ GitHub: Release build started
                  │
Monday 3:00 PM    │ ✅ Release created!
                  │
Monday 3:05 PM    │ Users: Download from Releases page
                  │ 🎉 Success!
```

## Cost Breakdown

### Free Tier (Public Repo):
- ✅ Unlimited builds
- ✅ Each build: ~120 minutes (2 platforms × 60 min)
- ✅ No cost!

### Free Tier (Private Repo):
- 2,000 minutes/month included
- Each build: ~120 minutes
- Can do: ~16 builds/month
- After that: $0.008/minute

### Optimization:
- Build only on tags (not every push) to save minutes
- Use branch filters in workflow
- Disable macOS build if you only need Windows

## Support & Resources

- 📖 Workflow file: `.github/workflows/build-executables.yml`
- 📖 Setup guide: `GITHUB_ACTIONS_SETUP.md`
- 📖 Workflow docs: `.github/workflows/README.md`
- 🔧 GitHub Actions: https://docs.github.com/en/actions
- 🔧 Nuitka: https://nuitka.net/

## Next Steps

1. ✅ Files created (workflow, scripts, docs)
2. ⏳ Push to GitHub
3. ⏳ First build (~60 min)
4. ⏳ Test executables
5. ⏳ Create release tag
6. ✅ Share with users!
