"""
recorder.py

Launches Chromium via Playwright's ASYNC API (required inside Kaggle/Jupyter,
which already runs its own asyncio event loop - the sync API raises
"using Playwright Sync API inside the asyncio loop").

Loads the self-contained HTML from renderer.py, lets the animation play,
and uses Playwright's built-in video recording to capture a .webm file.
Waits for window.__SCENE_DONE__ (set by templates/animation.js) instead of
a hardcoded sleep, so it never cuts off a scene early or waits too long.
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
GENERATED_VIDEO_DIR = BASE_DIR / "generated" / "videos"

DEFAULT_VIEWPORT = {"width": 1080, "height": 1920}  # portrait, short-form video


async def record_html(
    html_path: Path,
    run_id: str,
    viewport: dict = None,
    poll_interval: float = 0.5,
    max_wait_seconds: float = 180.0,
) -> Path:
    viewport = viewport or DEFAULT_VIEWPORT
    GENERATED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(
            viewport=viewport,
            record_video_dir=str(GENERATED_VIDEO_DIR),
            record_video_size=viewport,
        )
        page = await context.new_page()
        await page.goto(html_path.resolve().as_uri())

        elapsed = 0.0
        done = False
        while elapsed < max_wait_seconds:
            done = await page.evaluate("() => window.__SCENE_DONE__ === true")
            if done:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        if not done:
            print(
                f"[recorder] WARNING: scene did not signal completion within "
                f"{max_wait_seconds}s, stopping recording anyway (run_id={run_id})"
            )

        video = page.video  # must grab reference before closing
        await context.close()
        await browser.close()

        if video is None:
            raise RuntimeError("Playwright did not produce a video object")

        raw_path = Path(await video.path())
        final_path = GENERATED_VIDEO_DIR / f"{run_id}.webm"
        raw_path.rename(final_path)
        return final_path