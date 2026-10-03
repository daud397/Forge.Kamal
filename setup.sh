#!/usr/bin/env bash
# Listing Forge - one-shot server build for a fresh Ubuntu 22.04 / 24.04 VPS.
#
#   unzip listing-forge.zip -d /opt
#   bash /opt/listing-forge/setup.sh yourdomain.com
#
# Installs system packages, creates a service account that cannot log in,
# builds a virtualenv, writes a systemd unit that survives reboots, puts Nginx
# in front, enables the firewall and requests a Let's Encrypt certificate.
# Safe to re-run: existing .env values are kept, the password is not replaced.
set -euo pipefail

DOMAIN="${1:-}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE=forge
APP_USER=forge
PORT=8000

if [[ $EUID -ne 0 ]]; then
  echo "Run this as root (sudo bash $0 yourdomain.com)." >&2
  exit 1
fi
if [[ -z "$DOMAIN" ]]; then
  echo "Usage: bash $0 yourdomain.com" >&2
  exit 1
fi

echo "==> System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q python3 python3-venv python3-pip nginx certbot \
  python3-certbot-nginx ufw libgl1 libglib2.0-0 openssl

echo "==> Service account"
if ! id "$APP_USER" >/dev/null 2>&1; then
  useradd --system --no-create-home --shell /usr/sbin/nologin "$APP_USER"
fi

echo "==> Virtualenv"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q --upgrade pip
"$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

echo "==> .env"
ENV_FILE="$APP_DIR/.env"
[[ -f "$ENV_FILE" ]] || cp "$APP_DIR/.env.example" "$ENV_FILE"

# Set KEY=VALUE only when the key is missing or empty, so a re-run never
# replaces a password or secret people already use.
set_default() {
  local key="$1" value="$2"
  if grep -qE "^${key}=.+" "$ENV_FILE"; then
    return
  fi
  sed -i -E "/^#?\s*${key}=.*/d" "$ENV_FILE"
  echo "${key}=${value}" >> "$ENV_FILE"
}

NEW_PASSWORD="$(openssl rand -base64 18 | tr -d '/+=' | cut -c1-20)"
had_password=0
grep -qE "^PORTAL_PASSWORD=.+" "$ENV_FILE" && had_password=1
set_default PORTAL_PASSWORD "$NEW_PASSWORD"
set_default SESSION_SECRET "$(openssl rand -hex 32)"
set_default HTTPS 1
# Nginx on this box is the only proxy, and uvicorn already trusts 127.0.0.1,
# so the socket address is the real client.
set_default TRUSTED_PROXY_HOPS 0

chown root:root "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo "==> Data directory"
mkdir -p "$APP_DIR/data"
chown -R "$APP_USER:$APP_USER" "$APP_DIR/data"

echo "==> systemd unit"
cat > "/etc/systemd/system/${SERVICE}.service" <<UNIT
[Unit]
Description=Listing Forge
After=network-online.target
Wants=network-online.target

[Service]
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${APP_DIR}
# systemd reads this as root, so .env can stay readable by root only.
EnvironmentFile=${ENV_FILE}
ExecStart=${APP_DIR}/.venv/bin/uvicorn main:app --host 127.0.0.1 --port ${PORT} --proxy-headers --forwarded-allow-ips 127.0.0.1
Restart=on-failure
RestartSec=3
NoNewPrivileges=true
ProtectSystem=full
ReadWritePaths=${APP_DIR}/data

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now "$SERVICE"
systemctl restart "$SERVICE"

echo "==> Nginx"
cat > "/etc/nginx/sites-available/${SERVICE}" <<NGINX
server {
    listen 80;
    server_name ${DOMAIN};

    # Full-resolution photos are large; the app itself refuses over 80 MB a file.
    client_max_body_size 200m;

    location / {
        proxy_pass http://127.0.0.1:${PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        # A final-quality generation can take a couple of minutes.
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
NGINX
ln -sf "/etc/nginx/sites-available/${SERVICE}" "/etc/nginx/sites-enabled/${SERVICE}"
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

echo "==> Firewall"
ufw allow OpenSSH >/dev/null
ufw allow 'Nginx Full' >/dev/null
ufw --force enable >/dev/null

echo "==> Certificate"
if certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos \
     --register-unsafely-without-email --redirect; then
  echo "HTTPS is on."
else
  echo "Certificate step failed - usually the domain's A record does not point" >&2
  echo "here yet. Everything else works; rerun: certbot --nginx -d $DOMAIN" >&2
fi

echo
echo "Listing Forge is running at https://${DOMAIN}"
if [[ $had_password -eq 0 ]]; then
  echo
  echo "  Portal password: ${NEW_PASSWORD}"
  echo
  echo "Save it now. It is also in ${ENV_FILE} (root only)."
fi
echo "Add OPENAI_API_KEY to ${ENV_FILE}, then: systemctl restart ${SERVICE}"
