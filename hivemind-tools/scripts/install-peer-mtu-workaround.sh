#!/bin/sh
# Install a narrowly scoped IPv4 workaround for a diagnosed Tailscale MTU issue.
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo: sh $0 PEER_IPV4 [MTU]" >&2
    exit 1
fi
peer=${1:?Provide the affected peer IPv4 address}
mtu=${2:-1200}
python3 -c 'import ipaddress, sys; ipaddress.IPv4Address(sys.argv[1]); assert sys.argv[2].isdigit() and 576 <= int(sys.argv[2]) <= 1280' "$peer" "$mtu"
command -v systemctl >/dev/null
command -v ip >/dev/null
unit="glmbench-peer-mtu-$(printf '%s' "$peer" | tr . -)"

install -d -m 755 /usr/local/libexec
cat > /usr/local/libexec/glmbench-peer-mtu <<'HELPER'
#!/bin/sh
set -eu
peer=$1
mtu=$2
# Only adjust an existing Tailscale-owned peer route. Never create a route
# while Tailscale is down, or change another interface/default route.
route=$(ip -4 route show table 52 exact "$peer/32" 2>/dev/null || true)
case "$route" in
    *"dev tailscale0"*) ;;
    *) exit 0 ;;
esac
case "$route" in
    *"mtu $mtu"*) exit 0 ;;
esac
ip -4 route replace "$peer/32" dev tailscale0 mtu "$mtu" table 52
HELPER
chmod 755 /usr/local/libexec/glmbench-peer-mtu

cat > "/etc/systemd/system/$unit.service" <<UNIT
[Unit]
Description=GLM packet size workaround for $peer
After=tailscaled.service
ConditionPathExists=/sys/class/net/tailscale0

[Service]
Type=oneshot
ExecStart=/usr/local/libexec/glmbench-peer-mtu $peer $mtu
UNIT
cat > "/etc/systemd/system/$unit.timer" <<UNIT
[Unit]
Description=Restore GLM peer route after Tailscale reconnects

[Timer]
OnBootSec=10s
OnUnitActiveSec=30s
AccuracySec=5s
Unit=$unit.service

[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now "$unit.timer"
systemctl start "$unit.service"
printf 'Applied MTU %s for Tailscale peer %s only.\n' "$mtu" "$peer"
printf 'Disable: sudo systemctl disable --now %s.timer\n' "$unit"
printf 'Restore the current peer route: sudo ip -4 route replace %s/32 dev tailscale0 table 52\n' "$peer"
