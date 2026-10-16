#!/usr/bin/env bash
# Grant a user passwordless sudo on a remote Mac via a validated /etc/sudoers.d drop-in.
#   tools/grant-sudo.sh <host> [user]        e.g. tools/grant-sudo.sh my-mac.local
# Prompts once for the remote user's password (sudo); afterwards `sudo -n` works without one.
# Undo on the host:  sudo rm /etc/sudoers.d/90-<user>-nopasswd
set -euo pipefail

host="${1:?usage: $0 <host> [user]}"
user="${2:-$USER}"
file="/etc/sudoers.d/90-${user}-nopasswd"

# SSH_KEY=<path> pins a key (bypassing the agent); unset uses the normal ssh config/agent.
key="${SSH_KEY:-}"
ssh_opts=()
[[ -f "$key" ]] && ssh_opts=(-o IdentitiesOnly=yes -o IdentityAgent=none -i "$key")

# Sent as the ssh command (not on stdin) so the remote sudo can prompt on the terminal that -t allocates.
remote=$(cat <<EOF
set -euo pipefail
# macOS ships '#includedir /private/etc/sudoers.d' in /etc/sudoers; without it the drop-in would be ignored.
sudo grep -Eq '^[#@]includedir /(private/)?etc/sudoers.d' /etc/sudoers || { echo "error: /etc/sudoers does not include sudoers.d" >&2; exit 1; }
tmp=\$(mktemp)
trap 'rm -f "\$tmp"' EXIT
printf '%s\n' '${user} ALL=(ALL) NOPASSWD: ALL' > "\$tmp"
sudo visudo -cf "\$tmp"                        # syntax-check before installing; a broken sudoers file locks out sudo
sudo install -m 0440 -o root -g wheel "\$tmp" '${file}'
sudo visudo -c >/dev/null                      # re-check the full configuration with the drop-in in place
sudo -k                                        # drop the cached credential so the next line proves NOPASSWD works
sudo -n true && echo "passwordless sudo OK for ${user} on \$(hostname)"
EOF
)

ssh -t "${ssh_opts[@]}" "${user}@${host}" "bash -c $(printf '%q' "$remote")"
