#!/usr/bin/env python3
"""
Podcast Workflow — Nordisk Nexus
---------------------------------
Records one video file (DJI camera app), uploads it to YouTube,
transcribes it in Norwegian using OpenAI Whisper, and publishes a
new episode card to podcast.html — fully automated.

Usage
-----
  python podcast_workflow.py \\
    --video  path/to/episode.mp4 \\
    --title  "Episode 1: Ola Nordmann om grønn tech" \\
    --guest  "Ola Nordmann" \\
    --role   "Grûnder, TechNord AS" \\
    --desc   "Vi snakker om hva som trengs for å lykkes i Norge." \\
    --topic  "CleanTech"

First-time YouTube setup (one-off, ~5 min)
------------------------------------------
  1. Go to https://console.cloud.google.com and create a project
  2. Enable "YouTube Data API v3"
  3. Create OAuth credentials → Desktop app → download JSON
  4. Save the file as  scripts/client_secrets.json
  On first run the script opens a browser for Google sign-in and saves
  a token so you never need to log in again.

Required env vars
-----------------
  OPENAI_API_KEY   — OpenAI key for Whisper transcription

What it does (in order)
------------------------
  1. Uploads the video to YouTube (Unlisted by default)
  2. Extracts audio with ffmpeg and transcribes in Norwegian via Whisper
  3. Builds a timestamped episode card and injects it into podcast.html
  4. Saves the full transcript to output/episode_NNN_transcript.txt
  5. Commits podcast.html and pushes to git automatically
"""

import argparse
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
SCRIPTS_DIR = Path(__file__).parent
TOKEN_PATH = SCRIPTS_DIR / ".youtube_token.pkl"
SECRETS_PATH = SCRIPTS_DIR / "client_secrets.json"

# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        sys.exit(
            "Error: ffmpeg is not installed.\n"
            "  Mac:    brew install ffmpeg\n"
            "  Linux:  sudo apt install ffmpeg"
        )


def extract_audio(video_path: Path, out_dir: Path) -> Path:
    """Extract audio as 16-kHz mono WAV (Whisper requirement)."""
    audio_path = out_dir / "audio_for_whisper.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{result.stderr}")
    return audio_path


def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    return float(json.loads(result.stdout).get("format", {}).get("duration", 0))


# ---------------------------------------------------------------------------
# YouTube upload
# ---------------------------------------------------------------------------

def get_youtube_client():
    """Return an authenticated YouTube API client, opening a browser on first use."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        sys.exit(
            "Error: Google API packages not installed.\n"
            "  Run: pip install -r scripts/requirements.txt"
        )

    if not SECRETS_PATH.exists():
        sys.exit(
            f"Error: {SECRETS_PATH} not found.\n"
            "  Download OAuth credentials from Google Cloud Console\n"
            "  (APIs & Services → Credentials → Create → Desktop app)\n"
            "  and save as scripts/client_secrets.json"
        )

    creds = None
    if TOKEN_PATH.exists():
        creds = pickle.loads(TOKEN_PATH.read_bytes())

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(SECRETS_PATH), YOUTUBE_SCOPES
            )
            print("Opening browser for YouTube authorization (one-time only)...")
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_bytes(pickle.dumps(creds))

    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(video_path: Path, title: str, description: str, privacy: str) -> str:
    """Upload video and return its YouTube URL."""
    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        sys.exit("Error: google-api-python-client not installed.")

    youtube = get_youtube_client()

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": "22",  # People & Blogs
        },
        "status": {"privacyStatus": privacy},
    }

    media = MediaFileUpload(str(video_path), chunksize=10 * 1024 * 1024, resumable=True)
    request = youtube.videos().insert(
        part=",".join(body.keys()), body=body, media_body=media
    )

    print(f"Uploading to YouTube ({video_path.stat().st_size // (1024*1024)} MB)...")
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"  {pct}%", end="\r", flush=True)

    video_id = response["id"]
    url = f"https://youtu.be/{video_id}"
    print(f"  Uploaded: {url}          ")
    return url


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(audio_path: Path, api_key: str) -> dict:
    """Transcribe audio in Norwegian with OpenAI Whisper."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Error: openai not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    print("Transcribing in Norwegian with Whisper...")

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="no",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    if hasattr(response, "model_dump"):
        return response.model_dump()
    return dict(response)


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def build_transcript_segments_html(segments: list) -> str:
    lines = []
    for seg in segments:
        start = format_timestamp(seg.get("start", 0))
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines.append(
            f'            <div class="transcript-segment">\n'
            f'              <span class="transcript-time">{start}</span>\n'
            f'              <span class="transcript-text">{text}</span>\n'
            f'            </div>'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def build_episode_card(
    episode_num: int,
    title: str,
    guest: str,
    role: str,
    desc: str,
    topic: str,
    youtube_url: str,
    duration_str: str,
    date_str: str,
    transcript_segments_html: str,
) -> str:
    # YouTube embed URL
    embed_url = ""
    if youtube_url:
        yt_match = re.search(r"(?:youtu\.be/|v=)([\w-]+)", youtube_url)
        if yt_match:
            embed_url = f"https://www.youtube.com/embed/{yt_match.group(1)}"

    # Tag style
    tag_class = "tag"
    if topic in {"Finansiering", "Funding", "Events", "Arrangementer"}:
        tag_class = "tag tag-warm"
    elif topic in {"Kultur", "Culture", "Strategi", "Opinion"}:
        tag_class = "tag tag-surface"

    # Video block
    if embed_url:
        video_inner = f'<iframe src="{embed_url}" allowfullscreen title="{title}"></iframe>'
    else:
        video_inner = (
            '<div class="episode-video-placeholder">'
            '<span class="placeholder-icon">▶️</span>'
            '<span>Video ikke tilgjengelig ennå</span>'
            '</div>'
        )

    initials = "".join(w[0].upper() for w in guest.split()[:2]) if guest else "G"

    return f"""
            <!-- Episode {episode_num}: {title} — added {date_str} -->
            <div class="episode-card" data-episode="{episode_num}">
              <div class="episode-header">
                <div class="episode-thumbnail" onclick="toggleVideo({episode_num})">
                  <span class="episode-num">Episode {episode_num}</span>
                  🎙️
                  <button class="play-btn" aria-label="Spill av episode">▶</button>
                </div>
                <div class="episode-meta-panel">
                  <div class="episode-top">
                    <span class="{tag_class}">{topic}</span>
                    <h3 class="episode-title">{title}</h3>
                    <p class="episode-desc">{desc}</p>
                  </div>
                  <div class="episode-bottom">
                    <div class="episode-info">
                      <div class="episode-guest">
                        <div class="guest-avatar">{initials}</div>
                        <div class="guest-info">
                          <strong>{guest}</strong>
                          <span>{role}</span>
                        </div>
                      </div>
                      <div class="episode-info-item">
                        <span class="episode-info-icon">🗓</span>
                        <span>{date_str}</span>
                      </div>
                      <div class="episode-info-item">
                        <span class="episode-info-icon">⏱</span>
                        <span>{duration_str}</span>
                      </div>
                    </div>
                    <div class="episode-actions">
                      <button class="action-btn" data-video-btn="{episode_num}" onclick="toggleVideo({episode_num})">▶ Se video</button>
                      <button class="action-btn" data-transcript-btn="{episode_num}" onclick="toggleTranscript({episode_num})">📄 Transkript</button>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Video player -->
              <div class="episode-video" id="video-{episode_num}">
                {video_inner}
              </div>

              <!-- Transcript -->
              <div class="episode-transcript" id="transcript-{episode_num}">
                <div class="transcript-header">
                  <h4>Transkript — {title}</h4>
                  <button class="transcript-copy-btn" onclick="copyTranscript({episode_num})">Kopier</button>
                </div>
                <div class="transcript-body">
{transcript_segments_html}
                </div>
              </div>
            </div>"""


def inject_episode(podcast_html_path: Path, card_html: str, guest: str, role: str):
    html = podcast_html_path.read_text(encoding="utf-8")

    marker = '<!-- EPISODES ARE INJECTED HERE BY podcast_workflow.py -->'
    if marker not in html:
        sys.exit(f"Injection marker not found in {podcast_html_path}.")

    # Remove placeholder card
    html = re.sub(
        r'\s*<!-- Example episode card \(remove when real episodes are added\) -->.*?</div>\s*',
        "\n\n            ",
        html,
        flags=re.DOTALL,
    )

    html = html.replace(marker, marker + "\n" + card_html, 1)

    # Update guest widget
    initials = "".join(w[0].upper() for w in guest.split()[:2]) if guest else "G"
    new_guest = (
        f'<div class="guest-list-item">'
        f'<div class="guest-list-avatar">{initials}</div>'
        f'<div class="guest-list-info"><strong>{guest}</strong><span>{role}</span></div>'
        f'</div>'
    )
    placeholder = '<p style="font-size: 0.84rem; color: var(--text-muted);">Gjester vises her etter første episode.</p>'
    if placeholder in html:
        html = html.replace(placeholder, new_guest)
    else:
        html = html.replace(
            '</div>\n          </div>\n\n          <!-- Topics widget -->',
            f'\n              {new_guest}\n            </div>\n          </div>\n\n          <!-- Topics widget -->',
        )

    podcast_html_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_commit_and_push(podcast_html_path: Path, episode_num: int):
    repo_root = podcast_html_path.parent.parent
    rel_path = podcast_html_path.relative_to(repo_root)

    subprocess.run(["git", "add", str(rel_path)], cwd=repo_root, check=True)

    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=repo_root
    )
    if result.returncode == 0:
        print("No changes to commit (episode may already be published).")
        return

    subprocess.run(
        ["git", "commit", "-m", f"Add podcast episode {episode_num}"],
        cwd=repo_root, check=True,
    )
    subprocess.run(["git", "push"], cwd=repo_root, check=True)
    print("Pushed to git.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Upload DJI video to YouTube, transcribe in Norwegian, publish to podcast.html"
    )
    parser.add_argument("--video",    required=True, help="Path to video file (.mp4 / .mov)")
    parser.add_argument("--title",    required=True, help="Episode title")
    parser.add_argument("--guest",    required=True, help="Guest full name")
    parser.add_argument("--role",     default="Gjest", help="Guest role / company")
    parser.add_argument("--desc",     default="",   help="Short episode description")
    parser.add_argument("--topic",    default="Gründere", help="Topic tag")
    parser.add_argument("--privacy",  default="unlisted",
                        choices=["public", "unlisted", "private"],
                        help="YouTube privacy (default: unlisted)")
    parser.add_argument("--episode",  type=int, default=None, help="Episode number (auto-detected)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip YouTube upload (useful for testing)")
    parser.add_argument("--no-push",   action="store_true",
                        help="Skip git commit and push")
    parser.add_argument("--api-key",  default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument(
        "--podcast-html",
        default=str(SCRIPTS_DIR.parent / "olluri" / "podcast.html"),
    )
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("Error: Set OPENAI_API_KEY or pass --api-key.")

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Error: File not found: {video_path}")

    podcast_html = Path(args.podcast_html).resolve()
    if not podcast_html.exists():
        sys.exit(f"Error: podcast.html not found at {podcast_html}")

    check_ffmpeg()

    # Auto episode number
    if args.episode is None:
        existing = re.findall(r'data-episode="(\d+)"', podcast_html.read_text(encoding="utf-8"))
        args.episode = max((int(n) for n in existing), default=0) + 1

    print(f"\n--- Episode {args.episode}: {args.title} ---\n")

    # 1. YouTube upload
    youtube_url = ""
    if not args.no_upload:
        yt_description = (
            f"{args.desc}\n\n"
            f"Gjest: {args.guest} — {args.role}\n\n"
            f"Nordisk Nexus Podkasten"
        )
        youtube_url = upload_to_youtube(video_path, args.title, yt_description, args.privacy)
    else:
        print("Skipping YouTube upload (--no-upload).")

    # 2. Transcribe
    out_dir = SCRIPTS_DIR.parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("Extracting audio...")
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = extract_audio(video_path, Path(tmp))
        result = transcribe(audio_path, args.api_key)

    segments   = result.get("segments", [])
    full_text  = result.get("text", "").strip()

    ep_slug = f"episode_{args.episode:03d}"
    transcript_path = out_dir / f"{ep_slug}_transcript.txt"
    transcript_path.write_text(full_text, encoding="utf-8")
    print(f"Transcript saved → {transcript_path.name}")

    # 3. Duration
    secs = get_video_duration(video_path)
    duration_str = f"{math.floor(secs / 60)} min" if secs > 0 else "–"

    # Norwegian date, e.g. "9. april 2026"
    date_str = datetime.today().strftime("%-d. %B %Y")

    # 4. Build and inject HTML
    segments_html = build_transcript_segments_html(segments)
    card_html = build_episode_card(
        episode_num=args.episode,
        title=args.title,
        guest=args.guest,
        role=args.role,
        desc=args.desc,
        topic=args.topic,
        youtube_url=youtube_url,
        duration_str=duration_str,
        date_str=date_str,
        transcript_segments_html=segments_html,
    )
    inject_episode(podcast_html, card_html, args.guest, args.role)
    print(f"Episode {args.episode} injected into podcast.html")

    # 5. Git push
    if not args.no_push:
        git_commit_and_push(podcast_html, args.episode)

    print(f"\nDone! Episode {args.episode} is live.")
    if youtube_url:
        print(f"YouTube: {youtube_url}")


if __name__ == "__main__":
    main()
