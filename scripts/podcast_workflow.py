#!/usr/bin/env python3
"""
Podcast Workflow — Nordisk Nexus
---------------------------------
Records one video file (DJI camera app), transcribes it in Norwegian
using OpenAI Whisper, and publishes a new episode to podcast.html.

Usage
-----
  python podcast_workflow.py \\
    --video  path/to/episode.mp4 \\
    --title  "Episode 3: Ola Nordmann om grønn tech" \\
    --guest  "Ola Nordmann" \\
    --role   "Grûnder, TechNord AS" \\
    --desc   "Vi snakker om hva som trengs for å lykkes med grønn teknologi i Norge." \\
    --topic  "CleanTech" \\
    --youtube "https://youtu.be/XXXXXXXXXX"

Required
--------
  OPENAI_API_KEY  environment variable (or pass --api-key)

What it does
------------
  1. Extracts audio from the video with ffmpeg (for Whisper)
  2. Transcribes the audio in Norwegian using OpenAI Whisper
  3. Generates a timestamped transcript
  4. Builds a new episode card (HTML) and prepends it to podcast.html
  5. Saves the transcript as a plain .txt file in output/
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        sys.exit("Error: ffmpeg is not installed. Install it from https://ffmpeg.org/download.html")


def extract_audio(video_path: Path, out_dir: Path) -> Path:
    """Extract audio track from the video as a 16-kHz mono WAV for Whisper."""
    audio_path = out_dir / "audio_for_whisper.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                # no video
        "-ar", "16000",       # 16 kHz (Whisper requirement)
        "-ac", "1",           # mono
        "-c:a", "pcm_s16le",  # uncompressed WAV
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg error:\n", result.stderr)
        sys.exit("Failed to extract audio from video.")
    return audio_path


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS string."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def get_video_duration(video_path: Path) -> float:
    """Return duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    data = json.loads(result.stdout)
    return float(data.get("format", {}).get("duration", 0))


def transcribe(audio_path: Path, api_key: str) -> dict:
    """Transcribe audio in Norwegian using OpenAI Whisper. Returns the verbose JSON response."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Error: openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    print("Transcribing audio in Norwegian with Whisper... (this may take a minute)")

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="no",          # Norwegian
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    # openai SDK returns a Transcription object; convert to dict
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return dict(response)


def build_transcript_segments_html(segments: list) -> str:
    """Build the <div class="transcript-segment"> blocks from Whisper segments."""
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
    full_transcript: str,
) -> str:
    """Return the full HTML block for one episode card."""

    # Derive YouTube embed URL from watch URL
    embed_url = ""
    if youtube_url:
        # Handle youtu.be/ID and youtube.com/watch?v=ID
        yt_match = re.search(r"(?:youtu\.be/|v=)([\w-]+)", youtube_url)
        if yt_match:
            vid_id = yt_match.group(1)
            embed_url = f"https://www.youtube.com/embed/{vid_id}"

    # Topic tag style
    tag_class = "tag"
    warm_topics = {"Finansiering", "Funding", "Events", "Arrangementer"}
    surface_topics = {"Kultur", "Culture", "Strategi", "Opinion"}
    if topic in warm_topics:
        tag_class = "tag tag-warm"
    elif topic in surface_topics:
        tag_class = "tag tag-surface"

    # Video block
    if embed_url:
        video_inner = f'<iframe src="{embed_url}" allowfullscreen title="{title}"></iframe>'
    else:
        video_inner = (
            '<div class="episode-video-placeholder">'
            '<span class="placeholder-icon">▶️</span>'
            '<span>Last opp videoen til YouTube og kjør skriptet på nytt med --youtube URL</span>'
            '</div>'
        )

    # Guest initials for avatar
    initials = "".join(w[0].upper() for w in guest.split()[:2]) if guest else "G"

    card = f"""
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

    return card


def inject_episode(podcast_html_path: Path, card_html: str, guest: str, role: str, episode_num: int):
    """Prepend the episode card inside #episodeList and update the guest widget."""
    html = podcast_html_path.read_text(encoding="utf-8")

    # --- Insert episode card at the TOP of #episodeList ---
    marker = '<!-- EPISODES ARE INJECTED HERE BY podcast_workflow.py -->'
    if marker not in html:
        sys.exit(f"Injection marker not found in {podcast_html_path}. Has the file been modified?")

    # Remove the placeholder "no episodes yet" card if it's still there
    placeholder_pattern = re.compile(
        r'\s*<!-- Example episode card \(remove when real episodes are added\) -->.*?</div>\s*',
        re.DOTALL,
    )
    html = placeholder_pattern.sub("\n\n            ", html)

    html = html.replace(marker, marker + "\n" + card_html, 1)

    # --- Update guest widget ---
    guest_placeholder = '<p style="font-size: 0.84rem; color: var(--text-muted);">Gjester vises her etter første episode.</p>'
    initials = "".join(w[0].upper() for w in guest.split()[:2]) if guest else "G"
    new_guest_item = (
        f'<div class="guest-list-item">'
        f'<div class="guest-list-avatar">{initials}</div>'
        f'<div class="guest-list-info"><strong>{guest}</strong><span>{role}</span></div>'
        f'</div>'
    )
    if guest_placeholder in html:
        html = html.replace(guest_placeholder, new_guest_item)
    else:
        # Append to existing guest list
        html = html.replace(
            '</div>\n          </div>\n\n          <!-- Topics widget -->',
            f'\n              {new_guest_item}\n            </div>\n          </div>\n\n          <!-- Topics widget -->',
        )

    podcast_html_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Publish a podcast episode: transcribe with Whisper and inject into podcast.html"
    )
    parser.add_argument("--video",   required=True, help="Path to DJI video file (.mp4 / .mov)")
    parser.add_argument("--title",   required=True, help="Episode title")
    parser.add_argument("--guest",   required=True, help="Guest full name")
    parser.add_argument("--role",    default="Gjest",  help="Guest role / company")
    parser.add_argument("--desc",    default="",   help="Short episode description (1-2 sentences)")
    parser.add_argument("--topic",   default="Gründere", help="Topic tag (e.g. CleanTech, AI, Finansiering)")
    parser.add_argument("--youtube", default="",   help="YouTube URL after you upload the video")
    parser.add_argument("--episode", type=int, default=None, help="Episode number (auto-detected if omitted)")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument(
        "--podcast-html",
        default=str(Path(__file__).parent.parent / "olluri" / "podcast.html"),
        help="Path to podcast.html (default: ../olluri/podcast.html)",
    )
    args = parser.parse_args()

    # Validate
    if not args.api_key:
        sys.exit("Error: OpenAI API key required. Set OPENAI_API_KEY or pass --api-key.")

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Error: Video file not found: {video_path}")

    podcast_html = Path(args.podcast_html).resolve()
    if not podcast_html.exists():
        sys.exit(f"Error: podcast.html not found at {podcast_html}")

    check_ffmpeg()

    # Auto episode number
    if args.episode is None:
        existing = re.findall(r'data-episode="(\d+)"', podcast_html.read_text(encoding="utf-8"))
        args.episode = max((int(n) for n in existing), default=0) + 1

    print(f"Publishing episode {args.episode}: {args.title}")

    # Output directory
    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)

    # 1. Extract audio
    print("Extracting audio from video...")
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = extract_audio(video_path, Path(tmp))

        # 2. Transcribe
        result = transcribe(audio_path, args.api_key)

    segments = result.get("segments", [])
    full_text = result.get("text", "").strip()

    # 3. Save transcript
    ep_slug = f"episode_{args.episode:03d}"
    transcript_path = out_dir / f"{ep_slug}_transcript.txt"
    transcript_path.write_text(full_text, encoding="utf-8")
    print(f"Transcript saved to: {transcript_path}")

    # 4. Build duration string
    duration_secs = get_video_duration(video_path)
    if duration_secs > 0:
        mins = math.floor(duration_secs / 60)
        duration_str = f"{mins} min"
    else:
        duration_str = "–"

    date_str = datetime.today().strftime("%-d. %B %Y").lower()
    # Capitalise month
    date_str = date_str[0].upper() + date_str[1:]

    # 5. Build HTML
    segments_html = build_transcript_segments_html(segments)
    card_html = build_episode_card(
        episode_num=args.episode,
        title=args.title,
        guest=args.guest,
        role=args.role,
        desc=args.desc,
        topic=args.topic,
        youtube_url=args.youtube,
        duration_str=duration_str,
        date_str=date_str,
        transcript_segments_html=segments_html,
        full_transcript=full_text,
    )

    # 6. Inject into podcast.html
    inject_episode(podcast_html, card_html, args.guest, args.role, args.episode)
    print(f"Episode {args.episode} injected into {podcast_html}")

    # Done
    print()
    print("Done! Next steps:")
    print(f"  1. Upload {video_path.name} to YouTube (if you haven't yet)")
    if not args.youtube:
        print(f"  2. Re-run with --youtube <URL> to embed the video player")
    print(f"  3. git add olluri/podcast.html && git commit -m 'Add episode {args.episode}' && git push")


if __name__ == "__main__":
    main()
