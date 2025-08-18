set -euo pipefail
VER=0.7.3
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) PKG="mprocs-${VER}-linux-x86_64-musl.tar.gz" ;;
  aarch64|arm64) PKG="mprocs-${VER}-linux-aarch64-musl.tar.gz" ;;
  *) echo "unsupported arch: $ARCH"; exit 1 ;;
esac

curl -fL -o "$PKG" "https://github.com/pvolok/mprocs/releases/download/v${VER}/${PKG}"
tar -xzf "$PKG"              # extracts a single 'mprocs' binary
install -m 0755 mprocs /usr/local/bin/mprocs
rm -f "$PKG" mprocs
mprocs --version
