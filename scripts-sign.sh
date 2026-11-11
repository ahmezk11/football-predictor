#!/bin/bash
# Ad-hoc sign the app bundle.
#
# Why this exists: electron-builder rewrites Contents/Resources after the
# linker's own ad-hoc signature, which invalidates it. `spctl` then reports
# "code has no resources but signature indicates they must be present", and a
# copy carrying the quarantine flag is refused outright rather than showing the
# usual "unidentified developer" prompt. Signing here gives the bundle one
# consistent signature under its own identifier.
#
# Why it works in a temp directory: this project sits on an iCloud-synced
# Desktop, and the sync agent reattaches Finder metadata *while* codesign walks
# the bundle - "resource fork, Finder information, or similar detritus not
# allowed" - which no amount of clearing beforehand reliably beats. Signing on
# local disk and copying the result back does.
#
# This is NOT notarization. For a build other people download you need a
# Developer ID certificate and `xcrun notarytool submit`; see the README.
#
# With --stage the signed copy is left on local disk and its path printed on
# stdout, so the disk image can be built from a bundle that never touched the
# synced folder - copying it back reattaches the same metadata and leaves the
# .dmg carrying a signature that fails strict verification.
set -euo pipefail
APP="${1:?usage: scripts-sign.sh <path to .app> [--stage]}"
STAGE="${2:-}"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"

# Progress goes to stderr so --stage can print just the path.
log() { echo "$@" >&2; }

name="$(basename "$APP")"
work="$(mktemp -d)"
[ "$STAGE" = "--stage" ] || trap 'rm -rf "$work"' EXIT

log "signing $name"
cp -R "$APP" "$work/"
target="$work/$name"
xattr -cr "$target"

# Inside out: frameworks and helpers, then the frozen Python backend, then the
# bundle. Signing the outside first would only be invalidated by the inside.
for nested in "$target/Contents/Frameworks"/*.framework "$target/Contents/Frameworks"/*.app; do
  [ -e "$nested" ] && codesign --force --sign - "$nested" >/dev/null 2>&1
done
find "$target/Contents/Resources/backend" -type f \( -name "*.so" -o -name "*.dylib" \) \
  -exec codesign --force --sign - {} \; >/dev/null 2>&1 || true
codesign --force --sign - "$target/Contents/Resources/backend/predictor-server" >/dev/null 2>&1
codesign --force --sign - "$target"

codesign --verify --strict --verbose=2 "$target" 2>&1 | tail -2 >&2
codesign -dv "$target" 2>&1 | grep -E "Identifier=|Signature=" >&2 || true

# ditto replaces the bundle in one pass. Deleting and re-copying while
# electron-builder is still assembling the directory duplicated files, which is
# why this runs after packaging rather than from an afterPack hook.
ditto "$target" "$APP"
log "signed and copied back to $APP"
[ "$STAGE" = "--stage" ] && echo "$target"   # for building the disk image
exit 0
