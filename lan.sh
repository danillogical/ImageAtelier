# Sourced by setup.sh and run.sh: LAN name and addresses of this Mac.
lan_host() {
  local h
  h="$(scutil --get LocalHostName 2>/dev/null || hostname -s)"
  echo "${h}.local"
}
lan_ips() {
  # Every active IPv4 address on every interface (Wi-Fi, USB/Thunderbolt Ethernet, ...), loopback excluded.
  local i ip
  for i in $(ifconfig -l); do
    [[ "$i" == lo* ]] && continue
    ip="$(ipconfig getifaddr "$i" 2>/dev/null || true)"
    [[ -n "$ip" ]] && echo "$ip"
  done
  return 0
}
