# Deployment

## Services
- open-webui: Web UI
- oidc-discovery: Local OIDC discovery JSON for YRU Passport
- userinfo-proxy: Proxies userinfo + synthesizes email from JWT sub

## Setup
1. Copy `docker-compose.yml` to server
2. Set `OAUTH_CLIENT_SECRET` in environment
3. `docker compose up -d`
