# Icon Generation Instructions

Tauri requires icons in multiple formats. You need to create the following icon files in `src-tauri/icons/`:

## Required Icons

1. **32x32.png** - 32x32 pixels PNG
2. **128x128.png** - 128x128 pixels PNG
3. **128x128@2x.png** - 256x256 pixels PNG (for retina displays)
4. **icon.icns** - macOS icon (if building for macOS)
5. **icon.ico** - Windows icon

## Quick Generation

### Option 1: Use Tauri Icon Generator (Recommended)

Install the Tauri icon generator:
```bash
npm install --save-dev @tauri-apps/cli
```

Then generate icons from a single 1024x1024 PNG source image:
```bash
npx @tauri-apps/cli icon path/to/your-icon.png
```

This will automatically generate all required icon sizes.

### Option 2: Manual Creation

1. Create a 1024x1024 PNG image with your desired icon design
2. Use an online tool like [favicon.io](https://favicon.io/) or [icoconvert.com](https://icoconvert.com/)
3. Generate the required sizes and formats
4. Place them in `src-tauri/icons/`

### Option 3: Use Default Tauri Icons (Temporary)

For development, you can use Tauri's default icons:
```bash
cd src-tauri
npx @tauri-apps/cli icon
```

This will generate default Tauri icons.

## Icon Design Tips

- Use a simple, recognizable design
- Ensure it looks good at small sizes (32x32)
- Use high contrast colors
- Consider a robot/AI theme for the RNN server
- Make sure it's distinguishable from other apps

## Current Status

⚠️ **Icons need to be generated before building the app.**

Run one of the options above to create the required icon files.

