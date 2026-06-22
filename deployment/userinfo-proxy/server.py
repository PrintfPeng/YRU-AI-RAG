#!/usr/bin/env python3
"""
YRU Passport userinfo proxy.
- Proxies /oauth/token to capture id_token (if ever returned in future)
- Falls back to decoding access_token JWT
- Synthesizes email from sub when no email claim exists
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, base64, logging, urllib.request, urllib.error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
PASSPORT_TOKEN    = "https://passport.yru.ac.th/oauth/token"
PASSPORT_USERINFO = "https://passport.yru.ac.th/oidc/userinfo"
TOKEN_CLAIMS = {}   # access_token -> claims from id_token

def decode_jwt(token):
    parts = token.split(".")
    if len(parts) != 3:
        return None
    pad = parts[1] + "=" * (4 - len(parts[1]) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(pad))
    except Exception as e:
        logging.error("JWT decode: %s", e)
        return None

def enrich_claims(claims):
    """เพิ่ม email / preferred_username / name ถ้าขาด"""
    sub = str(claims.get("sub", ""))
    if sub and not claims.get("email"):
        claims["email"] = f"user{sub}@yru.ac.th"
        logging.info("Synthesized email: %s", claims["email"])
    if sub and not claims.get("preferred_username"):
        claims["preferred_username"] = f"user{sub}"
    if sub and not claims.get("name"):
        claims["name"] = f"YRU User {sub}"
    return claims

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logging.info("%s - " + fmt, self.address_string(), *args)

    # ── POST /oauth/token : proxy + แคช id_token ─────────────────────────
    def do_POST(self):
        if self.path != "/oauth/token":
            self.send_error(404); return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        fwd = {"Content-Type": self.headers.get("Content-Type",
                               "application/x-www-form-urlencoded")}
        auth = self.headers.get("Authorization", "")
        if auth:
            fwd["Authorization"] = auth
        try:
            req = urllib.request.Request(PASSPORT_TOKEN, data=body,
                                         headers=fwd, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp_body = resp.read()
            td = json.loads(resp_body)
            logging.info("Token keys: %s", list(td.keys()))

            at = td.get("access_token")
            it = td.get("id_token")
            if it and at:
                c = decode_jwt(it)
                if c:
                    TOKEN_CLAIMS[at] = enrich_claims(c)
                    logging.info("Cached id_token sub=%s email=%s",
                                 c.get("sub"), c.get("email"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp_body)
        except Exception as e:
            logging.error("Token proxy: %s", e)
            self.send_error(502, str(e))

    # ── GET /oidc/userinfo ────────────────────────────────────────────────
    def do_GET(self):
        if self.path != "/oidc/userinfo":
            self.send_error(404); return
        auth  = self.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else None
        if not token:
            self.send_error(401, "No Bearer token"); return

        # 1. ใช้ id_token claims ที่แคชไว้ (ถ้ามี)
        if token in TOKEN_CLAIMS:
            claims = TOKEN_CLAIMS[token]
            self._ok(claims, "cached id_token"); return

        # 2. ลอง passport userinfo ตรงๆ
        try:
            req = urllib.request.Request(
                PASSPORT_USERINFO,
                headers={"Authorization": "Bearer " + token,
                         "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read()
            c = json.loads(body)
            self._ok(enrich_claims(c), "passport userinfo"); return
        except urllib.error.HTTPError as e:
            logging.warning("Passport userinfo HTTP %s", e.code)
        except Exception as e:
            logging.warning("Passport userinfo: %s", e)

        # 3. decode access_token JWT แล้ว synthesize email
        c = decode_jwt(token)
        if c:
            self._ok(enrich_claims(c), "JWT+synthetic-email"); return

        self.send_error(500, "Cannot get user info")

    def _ok(self, claims, source):
        logging.info("Userinfo [%s] sub=%s email=%s name=%s",
                     source, claims.get("sub"), claims.get("email"), claims.get("name"))
        body = json.dumps(claims).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 80), Handler).serve_forever()
