"""Minimal EUMETSAT Data Store client (search + download), credentials from .env
(KEY, SECRET). Usage: datastore.py COLLECTION START END OUTDIR"""
import base64, json, os, sys, time, urllib.parse, urllib.request

API = "https://api.eumetsat.int"
FDHSI = "EO:EUM:DAT:0662"        # FCI Level 1c normal resolution, full disc, 0 deg
LFL = "EO:EUM:DAT:0691"          # LI Level 2 lightning flashes
HERE = os.path.dirname(os.path.abspath(__file__))


def credentials():
    """EUMETSAT_KEY / EUMETSAT_SECRET from the environment (CI secrets),
    else KEY / SECRET from .env next to this file."""
    if os.environ.get("EUMETSAT_KEY") and os.environ.get("EUMETSAT_SECRET"):
        return os.environ["EUMETSAT_KEY"], os.environ["EUMETSAT_SECRET"]
    env = dict(l.strip().split("=", 1) for l in open(os.path.join(HERE, ".env")) if "=" in l and not l.startswith("#"))
    return env["KEY"], env["SECRET"]


def token():
    key, secret = credentials()
    auth = base64.b64encode(f"{key}:{secret}".encode()).decode()
    req = urllib.request.Request(API + "/token", data=b"grant_type=client_credentials",
                                 headers={"Authorization": "Basic " + auth,
                                          "Content-Type": "application/x-www-form-urlencoded"})
    return json.load(urllib.request.urlopen(req, timeout=60))["access_token"]


def search(coll, start, end, n=500):
    q = urllib.parse.urlencode({"pi": coll, "dtstart": start, "dtend": end, "format": "json", "c": n})
    r = json.load(urllib.request.urlopen(f"{API}/data/search-products/1.0.0/os?{q}", timeout=180))
    return [f["id"] for f in r.get("features", [])]


def download(coll, pid, out_dir, tok):
    path = os.path.join(out_dir, pid + ".zip")
    if os.path.exists(path):
        return path
    url = (f"{API}/data/download/1.0.0/collections/{urllib.parse.quote(coll, safe='')}"
           f"/products/{urllib.parse.quote(pid, safe='')}")
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
    t0, n = time.time(), 0
    with urllib.request.urlopen(req, timeout=300) as r, open(path + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            n += len(chunk)
    os.replace(path + ".part", path)
    print(f"ok {pid[:60]}... {n / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
    return path


if __name__ == "__main__":
    coll, start, end, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    tok = token()
    ids = search(coll, start, end)
    print(len(ids), "products", flush=True)
    for pid in ids:
        download(coll, pid, out, tok)
