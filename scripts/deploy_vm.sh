#!/usr/bin/env bash
# Deploy the app to an Ubuntu VM (Oracle Cloud Always Free) with Docker.
#
#   scripts/deploy_vm.sh <public-ip>            # first deploy or update
#   scripts/deploy_vm.sh <public-ip> restart    # restart the container (persistence test)
#
# Needs ~/.ssh/oracle_vm (its public half was pasted into the VM at creation) and .env with
# DATABASE_URL, QDRANT_URL, QDRANT_API_KEY, LLM_PROVIDER, LLM_API_KEY. Only those settings are
# copied to the VM, into a root-only file outside the image; .env itself never leaves this Mac.
set -euo pipefail

IP="${1:?usage: scripts/deploy_vm.sh <public-ip> [restart]}"
ACTION="${2:-deploy}"
KEY="$HOME/.ssh/oracle_vm"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "ubuntu@$IP")
cd "$(dirname "$0")/.."

if [[ "$ACTION" == "restart" ]]; then
  "${SSH[@]}" "sudo docker restart investigator && sudo docker ps --filter name=investigator"
  exit 0
fi

echo "== 1/5 install Docker and open port 80 in the VM firewall"
"${SSH[@]}" 'set -e
  if ! command -v docker >/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq docker.io >/dev/null
    sudo systemctl enable --now docker
  fi
  # Oracle Ubuntu images reject everything except SSH in iptables; allow HTTP and keep it after reboot.
  if ! sudo iptables -C INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null; then
    sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
    sudo apt-get install -y -qq iptables-persistent >/dev/null 2>&1 || true
    sudo netfilter-persistent save >/dev/null 2>&1 || true
  fi
  mkdir -p ~/app'

echo "== 2/5 copy the code"
COPYFILE_DISABLE=1 tar czf - --exclude='__pycache__' --exclude='*.egg-info' \
  Dockerfile pyproject.toml README.md .dockerignore src | "${SSH[@]}" 'cd ~/app && tar xzf -'

echo "== 3/5 write the runtime settings (root-only file, values not printed)"
.venv/bin/python - <<'PY' | "${SSH[@]}" 'sudo install -m 600 /dev/stdin /etc/investigator.env'
from dotenv import dotenv_values
v = dotenv_values(".env")
keys = ["DATABASE_URL", "QDRANT_URL", "QDRANT_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "EMBEDDING_MODEL"]
missing = [k for k in keys[:3] + ["LLM_PROVIDER", "LLM_API_KEY"] if not v.get(k)]
if missing:
    raise SystemExit(f"missing in .env: {missing}")
for k in keys:
    if v.get(k):
        print(f"{k}={v[k]}")
print("CHUNK_STRATEGY=sentence")
print("INDEX_STRATEGIES=sentence")
PY

echo "== 4/5 build the image (first time about 5-10 minutes)"
"${SSH[@]}" 'cd ~/app && sudo docker build -q -t investigator . >/dev/null && echo built'

echo "== 5/5 (re)start the container; it restarts itself after crashes and reboots"
"${SSH[@]}" 'sudo docker rm -f investigator >/dev/null 2>&1 || true
  sudo docker run -d --name investigator --restart unless-stopped \
    --env-file /etc/investigator.env -p 80:7860 investigator >/dev/null
  sudo docker ps --filter name=investigator --format "{{.Names}} {{.Status}}"'

echo "App URL: http://$IP"
