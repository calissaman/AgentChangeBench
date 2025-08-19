set -euo pipefail

# Install mprocs
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

# Install Caddy for Debian
echo "Installing Caddy..."
apt-get update
apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
apt-get update
apt-get install -y caddy
caddy version
