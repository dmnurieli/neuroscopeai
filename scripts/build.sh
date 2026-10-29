#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
RUN=0; [[ "${1:-}" == "--run" ]] && RUN=1
(cd "$ROOT/engine" && cargo build --release)
(cd "$ROOT/app" && swift build -c release)
mkdir -p "$ROOT/build"
[[ -f "$ROOT/build/AppIcon.icns" ]] || swift "$ROOT/scripts/make-icon.swift"
APP="$ROOT/build/NeuroScope.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$ROOT/app/.build/release/NeuroScope" "$APP/Contents/MacOS/NeuroScope"
cp "$ROOT/app/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/build/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
[[ -f "$ROOT/docs/GUIDE.md" ]] && cp "$ROOT/docs/GUIDE.md" "$APP/Contents/Resources/GUIDE.md"
codesign --force --sign - "$APP"
echo "$APP"
[[ $RUN == 1 ]] && open "$APP"
exit 0
