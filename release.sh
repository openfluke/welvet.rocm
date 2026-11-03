#!/usr/bin/env bash
# Publish this independent AMD ROCm / HIP backend.
#
# Default / --source-only: commit local source, push branch + immutable tag,
# then create a GitHub Release (source tag; no binary build).
#
# Optional --package: also build/package Linux x86_64 shared+static artifacts
# (needs hipcc) and attach them to the release.
#
# GPU compute may be UNTESTED on the build host (no ROCm-capable device).
# The script never force-pushes or moves an existing tag.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
DRY_RUN=0
PLAN=0
SOURCE_ONLY=1
PACKAGE_BIN=0
SKIP_BUILD=0
SKIP_TESTS=0
TEST_GO=0
DRAFT=0
AUTO_COMMIT=1
BUILD_ROOT="$ROOT/build-release"
DIST="$ROOT/dist"

usage() {
  cat <<'HELP'
Usage: ./release.sh [options]
  --source-only   Tag + GitHub Release from source only (default; no hipcc package)
  --package       Also build/package linux-x86_64 shared+static binaries (needs hipcc)
  --dry-run       Do everything except commit/tag/push/publish
  --plan          Show version and actions only
  --skip-build    With --package: reuse build-release/shared and build-release/static
  --skip-tests    With --package: skip native ABI / GPU smoke checks
  --test-go       With --package: also run sibling w2a Mount suites
  --no-commit     Require a clean source tree instead of committing local changes
  --draft         Create a draft GitHub Release
  -h, --help      Show this help

Default: release v<current CMake/README/runtime version> as a source tag to origin.
Publishing requires an authenticated GitHub CLI (gh). No sibling repo is modified.
HELP
}
for arg in "$@"; do
  case "$arg" in
    --source-only) SOURCE_ONLY=1; PACKAGE_BIN=0 ;;
    --package) PACKAGE_BIN=1; SOURCE_ONLY=0 ;;
    --dry-run) DRY_RUN=1 ;;
    --plan) PLAN=1; DRY_RUN=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --test-go) TEST_GO=1 ;;
    --no-commit) AUTO_COMMIT=0 ;;
    --draft) DRAFT=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done
fail() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Required command missing: $1"; }
need git
need python3
VERSION="$(python3 - "$ROOT" <<'PY'
from pathlib import Path
import re,sys
r=Path(sys.argv[1])
cm=re.search(r'project\(welvet_rocm\s+VERSION\s+([0-9.]+)',(r/'CMakeLists.txt').read_text())
rm=re.search(r'\*\*Version\*\*\s*\|\s*\*\*`([0-9.]+)`',(r/'README.md').read_text())
abi=re.search(r'welvet_version\(void\)\s*\{.*?return\s+"([0-9.]+)"',(r/'src/interop/c_abi.cpp').read_text(),re.S)
if not all((cm,rm,abi)) or len({m[1] for m in (cm,rm,abi)})!=1:
    raise SystemExit('CMake, README and runtime source versions must agree')
print(cm[1])
PY
)"
SCORE="$(python3 - "$ROOT/README.md" <<'PY_SCORE'
from pathlib import Path
import re,sys
m=re.search(r'\*\*Engine board\*\*\s*\|\s*\*\*(\d+(?:\.\d+)?)\s*/\s*100',Path(sys.argv[1]).read_text())
if not m: raise SystemExit('README engine board not found')
print(m[1])
PY_SCORE
)"
TAG="v$VERSION"
BRANCH="$(git symbolic-ref --quiet --short HEAD)" || fail "Check out a branch before releasing"
REPO="$(git remote get-url origin | python3 -c '
import re,sys
url=sys.stdin.read().strip()
m=re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)([^/]+/[^/]+?)(?:\.git)?/?",url)
if not m: raise SystemExit("origin must point to a GitHub repository")
print(m[1])
')"
[[ "$REPO" == */welvet.rocm ]] || fail "origin must name a welvet.rocm repository"
if [[ "$DRY_RUN" -eq 0 ]]; then
  need gh
  gh auth status --hostname github.com >/dev/null 2>&1 || fail "Run gh auth login first"
  if [[ "$AUTO_COMMIT" -eq 0 && -n "$(git status --porcelain)" ]]; then
    fail "Source tree is dirty; commit changes or omit --no-commit"
  fi
  if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
    [[ "$(git rev-parse "$TAG^{commit}")" == "$(git rev-parse HEAD)" ]] || fail "$TAG already names another commit; tags are never moved"
    [[ -z "$(git status --porcelain)" ]] || fail "$TAG already exists; do not change the tagged source"
  fi
fi

KIND="source-only"
[[ "$PACKAGE_BIN" -eq 1 ]] && KIND="source + linux-x86_64 package"
printf 'welvet.rocm %s (%s)\nGitHub: %s\nBranch: %s\n' "$TAG" "$KIND" "$REPO" "$BRANCH"
if [[ "$PLAN" -eq 1 ]]; then
  if [[ "$PACKAGE_BIN" -eq 1 ]]; then
    echo "Plan: build shared + static -> ABI smoke (GPU untested OK if no device) -> package + SHA256"
    [[ "$TEST_GO" -eq 0 ]] || echo "Plan: test the staged .so through sibling Go/w2a"
  else
    echo "Plan: source tag + GitHub Release notes (no hipcc / no binary assets)"
  fi
  echo "Publish: commit local source (unless --no-commit), push branch + immutable tag, gh release create"
  echo "Plan only: no files, git state or remote state changed"
  exit 0
fi

mkdir -p "$DIST"
NOTES="$DIST/release-$TAG.md"
NATIVE_CHECK="not run (source-only release)"
GO_CHECK="not requested"
ARCHIVE=""
SHA=""

# ---- optional binary package (needs ROCm / hipcc) ----
if [[ "$PACKAGE_BIN" -eq 1 ]]; then
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail "--package requires Linux x86_64"
  need cmake
  need tar
  need sha256sum
  need readelf
  need hipcc

  if [[ -z "${ROCM_PATH:-}" ]]; then
    for cand in /usr /opt/rocm /opt/rocm-7.1.0; do
      if [[ -x "${cand}/bin/hipcc" ]] || [[ -x /usr/bin/hipcc && "$cand" == /usr ]]; then
        export ROCM_PATH="$cand"
        break
      fi
    done
  fi
  export PATH="${ROCM_PATH:-/usr}/bin:${PATH}"
  if [[ -d /usr/lib64/rocm/llvm/bin ]]; then
    export PATH="/usr/lib64/rocm/llvm/bin:${PATH}"
  fi

  JOBS="$(nproc 2>/dev/null || echo 4)"
  mkdir -p "$BUILD_ROOT"
  STAGE="$(mktemp -d "$DIST/.stage.XXXXXX")"
  cleanup() { python3 - "$STAGE" <<'PY'
from pathlib import Path
import shutil,sys
p=Path(sys.argv[1])
if p.name.startswith('.stage.'):
    shutil.rmtree(p,ignore_errors=True)
PY
  }
  trap cleanup EXIT
  NAME="welvet-rocm-$TAG-linux-x86_64"
  PACKAGE="$STAGE/$NAME"
  mkdir -p "$PACKAGE"

  CMAKE_COMMON=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/usr}"
    -DCMAKE_INSTALL_PREFIX="$PACKAGE"
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF
    -DWELVET_BUILD_EXAMPLES=OFF
  )
  if [[ -n "${CMAKE_HIP_ARCHITECTURES:-}" ]]; then
    CMAKE_COMMON+=(-DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES}")
  fi

  if [[ "$SKIP_BUILD" -eq 0 ]]; then
    for kind in shared static; do
      shared=ON
      tools=ON
      [[ "$kind" == shared ]] || { shared=OFF; tools=OFF; }
      cmake -S "$ROOT" -B "$BUILD_ROOT/$kind" \
        "${CMAKE_COMMON[@]}" \
        -DWELVET_BUILD_SHARED="$shared" -DWELVET_BUILD_TOOLS="$tools"
      cmake --build "$BUILD_ROOT/$kind" --parallel "$JOBS"
    done
  fi
  for artifact in shared/libwelvet_rocm.so shared/welvet_dense_fwd static/libwelvet_rocm.a; do
    [[ -f "$BUILD_ROOT/$artifact" ]] || fail "Missing $BUILD_ROOT/$artifact; run without --skip-build"
  done
  cmake --install "$BUILD_ROOT/shared" --prefix "$PACKAGE"
  cmake --install "$BUILD_ROOT/static" --prefix "$PACKAGE"
  # CMake may install as libwelvet.so — normalize name if needed
  if [[ ! -f "$PACKAGE/lib/libwelvet_rocm.so" && -f "$PACKAGE/lib/libwelvet.so" ]]; then
    mv "$PACKAGE/lib/libwelvet.so" "$PACKAGE/lib/libwelvet_rocm.so"
  fi
  if [[ ! -f "$PACKAGE/lib/libwelvet_rocm.a" && -f "$PACKAGE/lib/libwelvet.a" ]]; then
    mv "$PACKAGE/lib/libwelvet.a" "$PACKAGE/lib/libwelvet_rocm.a"
  fi
  [[ -f "$PACKAGE/lib/libwelvet_rocm.so" ]] || fail "Packaged shared library missing"
  [[ "$("$PACKAGE/bin/welvet_dense_fwd" --version)" == "$VERSION" ]] || fail "Packaged CLI version mismatch"
  if readelf -d "$PACKAGE/bin/welvet_dense_fwd" | python3 -c '
import sys
text=sys.stdin.read()
bad=False
for line in text.splitlines():
    if "RPATH" in line or "RUNPATH" in line:
        if "/home/" in line or "/build-release/" in line or "/build/" in line:
            bad=True
sys.exit(1 if bad else 0)
'; then
    fail "Packaged tool embeds a non-relocatable RPATH"
  fi

  NATIVE_CHECK="skipped (--skip-tests)"
  if [[ "$SKIP_TESTS" -eq 0 ]]; then
    smoke_out="$(python3 "$ROOT/scripts/release_smoke.py" "$PACKAGE/lib/libwelvet_rocm.so" "$VERSION")"
    echo "$smoke_out"
    if echo "$smoke_out" | grep -qi 'UNTESTED'; then
      NATIVE_CHECK="ABI/load passed; GPU compute UNTESTED (no ROCm-capable device on build host)"
    else
      NATIVE_CHECK="passed: packaged C ABI + ROCm identity + forward/backward/SGD"
    fi
  fi
  GO_CHECK="not requested"
  if [[ "$TEST_GO" -eq 1 ]]; then
    need go
    W2A="${WELVET_W2A_ROOT:-$ROOT/../w2a}"
    [[ -f "$W2A/go.mod" ]] || fail "Sibling w2a missing; set WELVET_W2A_ROOT or omit --test-go"
    (
      cd "$W2A"
      env -u CC -u CXX PATH="/usr/bin:/usr/local/go/bin:${PATH}" \
        WELVET_ROCM_LIBRARY="$PACKAGE/lib/libwelvet_rocm.so" \
        LD_LIBRARY_PATH="$PACKAGE/lib:${LD_LIBRARY_PATH:-}" \
        go test ./tests/go_v_rocm -count=1 -timeout 25m \
        -run '^(TestGoVRocmSmokeVersion|TestGoVRocmDenseForward)$'
    )
    GO_CHECK="ran: Go smoke/version + DenseForward (GPU cells may SKIP without HSA device)"
  fi

  cp "$ROOT/README.md" "$ROOT/VERSIONING.md" "$ROOT/LICENSE" "$PACKAGE/" 2>/dev/null || \
    cp "$ROOT/README.md" "$ROOT/VERSIONING.md" "$PACKAGE/"
  mkdir -p "$PACKAGE/docs"
  cp "$ROOT/docs/GO_INTEGRATION.md" "$ROOT/docs/ROCM_ONLY.md" "$PACKAGE/docs/" 2>/dev/null || true
  python3 - "$PACKAGE" "$VERSION" "PENDING" "false" "$NATIVE_CHECK" "$GO_CHECK" <<'PY'
import json,subprocess,sys
from pathlib import Path
p,v,commit,dirty,native,go=sys.argv[1:]
try:
    hip=subprocess.check_output(['hipcc','--version'],text=True).splitlines()[0]
except Exception:
    hip='unknown'
try:
    cxx=subprocess.check_output(['g++','--version'],text=True).splitlines()[0]
except Exception:
    cxx='unknown'
manifest=dict(engine='welvet.rocm',version=v,platform='linux-x86_64',
              model_abi=1,segment_extension=1,source_commit=commit,source_dirty_at_build=dirty=='true',
              native_validation=native,go_validation=go,hip_compiler=hip,compiler=cxx,
              gpu_compute='untested_unless_native_validation_says_passed')
Path(p,'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
  ARCHIVE="$DIST/$NAME.tar.gz"
  tar -czf "$ARCHIVE" -C "$STAGE" "$NAME"
  (cd "$DIST" && sha256sum "$NAME.tar.gz" > "$NAME.sha256")
  SHA="$DIST/$NAME.sha256"
fi

# ---- git commit / tag / notes ----
if [[ "$DRY_RUN" -eq 0 && -n "$(git status --porcelain)" ]]; then
  [[ "$AUTO_COMMIT" -eq 1 ]] || fail "Dirty source tree"
  git add -A
  git commit -m "Release welvet.rocm $TAG"
fi
COMMIT="$(git rev-parse HEAD)"
DIRTY=false
[[ -z "$(git status --porcelain)" ]] || DIRTY=true

cat > "$NOTES" <<NOTES
# welvet.rocm $TAG

Independent AMD ROCm / HIP backend for Welvet. Engine board: $SCORE/100;
see the tagged README and VERSIONING.md for the current subgate accounting.

- Supported numerical forward, backward and training on AMD GPU through HIP kernels (ROCm).
- 23 baseline layer families, 29 numbered training modes, Cameral/CamSync + mixed Go routing, 34 native parameter encodings where supported (GDN Float32).
- Persistent Go mount/forward/training/weight-return API (\`nativerocm\`); explicit JSON routing in the Go binding.
- Float32 activation interface; best-effort numerical compatibility targets the industry60–exact100 bands.
- Separate from welvet.cpp / welvet.cuda / welvet.metal; this package has no CUDA/Metal sibling runtime dependency.

## Validation status

**GPU compute: UNTESTED on the packaging host** unless notes below say otherwise.
Build host may lack a ROCm-capable HSA device (e.g. Polaris/Picasso). Linux x86_64
compile + C ABI / library load are validated; re-confirm exact_bit on a supported AMD GPU
(RDNA2/3 / Instinct per AMD ROCm matrix).

Validation: $NATIVE_CHECK. Go/w2a: $GO_CHECK.
Dirty working-tree package: $DIRTY.

## Source

Tag \`$TAG\` on branch \`$BRANCH\` (commit \`$COMMIT\`).
Clone and build with \`./scripts/build_linux.sh\` (needs ROCm \`hipcc\` / \`rocm-hip-devel\`).
Install into Go Welvet: \`./scripts/linux_install_go.sh\`.
NOTES

if [[ -n "$ARCHIVE" ]]; then
  cat >> "$NOTES" <<NOTES

## Linux x86_64 binary artifact

\`$(basename "$ARCHIVE")\` — shared + static libraries, headers, kernels, \`welvet_dense_fwd\`, docs, MANIFEST.
Verify with: \`sha256sum -c $(basename "$SHA")\`
NOTES
fi

printf '\nPrepared notes:\n  %s\n' "$NOTES"
[[ -z "$ARCHIVE" ]] || printf '  %s\n  %s\n' "$ARCHIVE" "$SHA"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete: no commit, tag, push or GitHub release performed"
  exit 0
fi

if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
  [[ "$(git rev-parse "$TAG^{commit}")" == "$COMMIT" ]] || fail "$TAG is not the release commit"
else
  git tag -a "$TAG" -m "welvet.rocm $TAG"
fi
git push origin "HEAD:refs/heads/$BRANCH"
git push origin "refs/tags/$TAG"

release_args=("$TAG" --repo "$REPO" --verify-tag --title "welvet.rocm $TAG" --notes-file "$NOTES")
[[ -z "$ARCHIVE" ]] || release_args+=("$ARCHIVE" "$SHA")
[[ "$DRAFT" -eq 0 ]] || release_args+=(--draft)
gh release create "${release_args[@]}"
echo "Release: https://github.com/$REPO/releases/tag/$TAG"
