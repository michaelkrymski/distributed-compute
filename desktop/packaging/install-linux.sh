#!/usr/bin/env sh
set -eu
src=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
dst="${XDG_DATA_HOME:-$HOME/.local/share}/distributed-compute"
mkdir -p "$dst" "$HOME/.local/share/applications"
cp -R "$src/Distributed-Compute/." "$dst/"
chmod +x "$dst/Distributed-Compute"
printf '[Desktop Entry]\nType=Application\nName=Distributed Compute\nExec="%s/Distributed-Compute"\nTerminal=false\nCategories=Science;\n' "$dst" > "$HOME/.local/share/applications/distributed-compute.desktop"
exec "$dst/Distributed-Compute"
