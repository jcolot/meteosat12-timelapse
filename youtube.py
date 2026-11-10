#!/usr/bin/env python3
"""Upload the day's videos to YouTube (YouTube Data API v3, standard library only).

  youtube.py auth
      One-time, on your machine: opens Google's consent page in the browser
      (scope youtube: upload, playlists, channel name) and stores the refresh token as the GitHub secret
      YOUTUBE_REFRESH_TOKEN via `gh secret set` (it is never printed).
      Needs YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET (env or .env): an OAuth
      client of type "Desktop app" in a Google Cloud project with the
      YouTube Data API v3 enabled.
  youtube.py upload FILE --title T [--description-file F] [--tags a,b] [--privacy public]
      [--playlist ID]   Resumable upload; prints the video URL; adds it to the playlist.
  youtube.py playlist-add PLAYLIST_ID VIDEO_ID [VIDEO_ID ...]
      Add already uploaded videos to a playlist.

Credentials for upload: YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN.
"""
import argparse, http.server, json, os, secrets, subprocess, sys, time, urllib.error, urllib.parse
import urllib.request, webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
# manage the channel's videos and playlists (upload alone cannot add to playlists)
SCOPE = "https://www.googleapis.com/auth/youtube"
AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN = "https://oauth2.googleapis.com/token"
UPLOAD = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
CHUNK = 64 << 20


def env(name):
    """From the environment, else from .env (NAME=value)."""
    if os.environ.get(name):
        return os.environ[name]
    p = os.path.join(HERE, ".env")
    if os.path.exists(p):
        for line in open(p):
            k, _, v = line.strip().partition("=")
            if k == name:
                return v.strip().strip('"\'')
    sys.exit(f"{name} is not set (environment or .env)")


def post_form(url, data):
    req = urllib.request.Request(url, urllib.parse.urlencode(data).encode())
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def channel(tok):
    """Title and handle of the channel this access token uploads to."""
    req = urllib.request.Request("https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",
                                 headers={"Authorization": "Bearer " + tok})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            items = json.load(r).get("items", [])
    except urllib.error.HTTPError as e:
        return f"(unknown: HTTP {e.code} {e.read()[:400].decode(errors='replace')})"
    if not items:
        return "(no channel)"
    sn = items[0]["snippet"]
    return f"{sn['title']} ({sn.get('customUrl', '?')})"


def playlist_add(tok, playlist, video):
    body = dict(snippet=dict(playlistId=playlist, resourceId=dict(kind="youtube#video", videoId=video)))
    req = urllib.request.Request("https://www.googleapis.com/youtube/v3/playlistItems?part=snippet",
                                 json.dumps(body).encode(), headers={
                                     "Authorization": "Bearer " + tok, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60):
            print(f"added {video} to playlist {playlist}", flush=True)
    except urllib.error.HTTPError as e:                  # the upload itself succeeded: report, go on
        print(f"could not add {video} to playlist {playlist}: HTTP {e.code} {e.read()[:400]!r}", flush=True)
        return False
    return True


def playlist_cmd(a):
    tok = access_token()
    if not all([playlist_add(tok, a.playlist, v) for v in a.videos]):
        sys.exit(1)


def auth(a):
    cid, csec = env("YOUTUBE_CLIENT_ID"), env("YOUTUBE_CLIENT_SECRET")
    state, got = secrets.token_urlsafe(16), {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if q.get("state", [""])[0] == state:
                got.update({k: v[0] for k, v in q.items()})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Done, you can close this tab.")

        def log_message(self, *_):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    redirect = f"http://127.0.0.1:{srv.server_port}"
    url = AUTH + "?" + urllib.parse.urlencode(dict(
        client_id=cid, redirect_uri=redirect, response_type="code", scope=SCOPE,
        access_type="offline", prompt="consent", state=state))
    print("Opening the consent page (if it does not open, visit):\n" + url, flush=True)
    webbrowser.open(url)
    while "code" not in got and "error" not in got:
        srv.handle_request()
    if "error" in got:
        sys.exit("consent refused: " + got["error"])
    tok = post_form(TOKEN, dict(code=got["code"], client_id=cid, client_secret=csec,
                                redirect_uri=redirect, grant_type="authorization_code"))
    if "refresh_token" not in tok:
        sys.exit("no refresh token returned")
    name = channel(tok["access_token"])
    if input(f"Token for channel {name}. Store it? [y/N] ").strip().lower() != "y":
        sys.exit("not stored")
    subprocess.run(["gh", "secret", "set", "YOUTUBE_REFRESH_TOKEN"], input=tok["refresh_token"].encode(),
                   check=True, cwd=HERE)
    print("Stored YOUTUBE_REFRESH_TOKEN as a repository secret.")


def access_token():
    return post_form(TOKEN, dict(client_id=env("YOUTUBE_CLIENT_ID"), client_secret=env("YOUTUBE_CLIENT_SECRET"),
                                 refresh_token=env("YOUTUBE_REFRESH_TOKEN"),
                                 grant_type="refresh_token"))["access_token"]


def upload(a):
    tok = access_token()
    print("channel:", channel(tok), flush=True)
    size = os.path.getsize(a.file)
    meta = dict(snippet=dict(title=a.title[:100], categoryId="28",          # Science & Technology
                             description=open(a.description_file).read()[:4900] if a.description_file else "",
                             tags=[t for t in (a.tags or "").split(",") if t]),
                status=dict(privacyStatus=a.privacy, selfDeclaredMadeForKids=False))
    req = urllib.request.Request(UPLOAD, json.dumps(meta).encode(), headers={
        "Authorization": "Bearer " + tok, "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)})
    with urllib.request.urlopen(req, timeout=60) as r:
        session = r.headers["Location"]
    sent, tries = 0, 0
    with open(a.file, "rb") as f:
        while True:
            f.seek(sent)
            body = f.read(CHUNK)
            end = sent + len(body) - 1
            req = urllib.request.Request(session, body, method="PUT", headers={
                "Authorization": "Bearer " + tok, "Content-Length": str(len(body)),
                "Content-Range": f"bytes {sent}-{end}/{size}"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    vid = json.load(r)["id"]                              # 200/201: done
                    break
            except urllib.error.HTTPError as e:
                if e.code == 308:                                         # chunk stored, continue
                    rng = e.headers.get("Range")
                    sent = int(rng.split("-")[1]) + 1 if rng else 0
                    print(f"  {sent / size:.0%}", flush=True)
                    continue
                if e.code < 500 or tries >= 5:
                    sys.exit(f"upload failed: HTTP {e.code} {e.read()[:500]!r}")
            except OSError:
                if tries >= 5:
                    raise
            tries += 1                                                    # 5xx / network: ask where we are
            time.sleep(10 * tries)
            q = urllib.request.Request(session, b"", method="PUT", headers={
                "Authorization": "Bearer " + tok, "Content-Length": "0", "Content-Range": f"bytes */{size}"})
            try:
                urllib.request.urlopen(q, timeout=60)
            except urllib.error.HTTPError as e:
                rng = e.headers.get("Range") if e.code == 308 else None
                sent = int(rng.split("-")[1]) + 1 if rng else 0
    print(f"https://youtu.be/{vid}", flush=True)
    if a.playlist:
        playlist_add(tok, a.playlist, vid)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("auth")
    u = sub.add_parser("upload")
    u.add_argument("file")
    u.add_argument("--title", required=True)
    u.add_argument("--description-file")
    u.add_argument("--tags")
    u.add_argument("--privacy", default="public", choices=["public", "unlisted", "private"])
    u.add_argument("--playlist", help="playlist ID to add the video to")
    pl = sub.add_parser("playlist-add")
    pl.add_argument("playlist")
    pl.add_argument("videos", nargs="+")
    a = p.parse_args()
    {"auth": auth, "upload": upload, "playlist-add": playlist_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
