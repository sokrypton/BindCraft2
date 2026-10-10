#!/usr/bin/env bash
# Re-copy the kernels from ColabDesign2, where they are developed.
#
# Vendoring means two copies, and two copies drift.  This makes the drift
# visible rather than silent: run it with --check in CI or by hand to see
# whether this copy is behind, and without a flag to update it.
#
#   bash native/sync.sh --check     # say what differs, change nothing
#   bash native/sync.sh             # copy over and record the new commit
set -euo pipefail
cd "$(dirname "$0")"

SRC="${COLABDESIGN2:-$HOME/ColabDesign2}/dev/native_block"
if [ ! -d "$SRC" ]; then
  echo "sync: no ColabDesign2 kernels at $SRC (set \$COLABDESIGN2)" >&2
  exit 1
fi

FILES=$(ls ./*.cu ./*.cuh | xargs -n1 basename)
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

drift=0
for f in $FILES; do
  if [ ! -f "$SRC/$f" ]; then
    echo "sync: $f is no longer in ColabDesign2"; drift=1; continue
  fi
  if ! cmp -s "$f" "$SRC/$f"; then
    echo "sync: $f differs"; drift=1
    [ "$CHECK" = 0 ] && cp "$SRC/$f" "$f"
  fi
done

if [ "$drift" = 0 ]; then
  echo "sync: in step with $SRC"
  exit 0
fi
if [ "$CHECK" = 1 ]; then
  echo "sync: this copy is OUT OF STEP -- run native/sync.sh to update"
  exit 1
fi
git -C "${COLABDESIGN2:-$HOME/ColabDesign2}" rev-parse --short HEAD > SOURCE_COMMIT
echo "sync: updated to $(cat SOURCE_COMMIT); rebuild with native/build.sh"
