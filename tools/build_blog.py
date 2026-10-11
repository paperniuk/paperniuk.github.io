#!/usr/bin/env python3
"""Build the blog from posts/*.md.

    pip install markdown
    python3 tools/build_blog.py

Writes blog/index.html, blog/<slug>/index.html, blog/feed.xml and sitemap.xml,
and refreshes the post list between the posts:start / posts:end markers in
index.html. Post images live in blog/img/<slug>/ and are referenced from the
markdown as img/<slug>/<file>.
"""
import html
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SITE = "https://paperniuk.github.io"
WORDS_PER_MINUTE = 200
HOME_POSTS = 3

ARROW_OUT = '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 7h10v10"/><path d="M7 17 17 7"/></svg>'
ARROW_IN = ARROW_OUT.replace('<path d="M7 7h10v10"/><path d="M7 17 17 7"/>', '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>')
FAVICON = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' fill='%2320242c'/%3E%3Ctext x='16' y='22' font-family='Helvetica,Arial,sans-serif' font-weight='600' font-size='17' fill='%239cb6d8' text-anchor='middle'%3EP%3C/text%3E%3C/svg%3E"


def esc(text):
    return html.escape(text, quote=True)


def parse_post(path):
    raw = path.read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n(.*)", raw, re.S)
    if not match:
        sys.exit(f"{path}: missing front matter")
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    for key in ("title", "date", "slug", "summary"):
        if not meta.get(key):
            sys.exit(f"{path}: front matter needs '{key}'")
    meta["body"] = match.group(2)
    meta["words"] = len(re.findall(r"\w+", meta["body"]))
    meta["minutes"] = max(1, round(meta["words"] / WORDS_PER_MINUTE))
    meta["when"] = datetime.strptime(meta["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    meta["tags"] = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
    return meta


def linkify(body):
    """Turn bare URLs into autolinks, leaving code and existing links alone."""
    out, in_fence = [], False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        parts = line.split("`")
        for i in range(0, len(parts), 2):  # even parts are outside inline code
            parts[i] = re.sub(
                r'(?<![(<\["\'])(https?://[^\s<>)\]]+?)([.,;:]*)(?=\s|$|\))',
                r"<\1>\2",
                parts[i],
            )
        out.append("`".join(parts))
    return "\n".join(out)


def png_size(path):
    with path.open("rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def render_body(meta):
    text = markdown.markdown(
        linkify(meta["body"]), extensions=["tables", "fenced_code", "sane_lists"]
    )
    text = re.sub(r"(<table>.*?</table>)", r'<div class="table-scroll">\1</div>', text, flags=re.S)

    return re.sub(r'<img alt="([^"]*)" src="([^"]+)" ?/?>', img_tag, text)


def img_tag(match):
    alt, src = match.group(1), match.group(2)
    size = png_size(ROOT / "blog" / src) if src.endswith(".png") else None
    dims = f' width="{size[0]}" height="{size[1]}"' if size else ""
    return f'<img src="../{src}" alt="{alt}"{dims} loading="lazy" decoding="async">'


def head(title, description, canonical, depth, extra=""):
    up = "../" * depth
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <meta name="description" content="{esc(description)}">
  <link rel="canonical" href="{esc(canonical)}">
  <link rel="alternate" type="application/atom+xml" title="Erik Paperniuk, notes" href="{SITE}/blog/feed.xml">
  <meta property="og:site_name" content="Erik Paperniuk">
  <meta property="og:title" content="{esc(title)}">
  <meta property="og:description" content="{esc(description)}">
  <meta property="og:url" content="{esc(canonical)}">
{extra}  <link rel="icon" href="{FAVICON}">
  <link rel="preload" href="{up}fonts/space-grotesk-var-latin.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="preload" href="{up}fonts/inter-var-latin.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="stylesheet" href="{up}assets/site.css">
</head>
<body>
  <a class="skip-link" href="#main-content">Skip to main content</a>
  <header class="site-header">
    <nav class="site-nav container" aria-label="Primary navigation">
      <a class="site-brand" href="{up}">Erik Paperniuk</a>
      <ul class="nav-links">
        <li><a href="{up}#work">Projects</a></li>
        <li><a href="{up}#research">Research</a></li>
        <li><a href="{up}#benchmarks">Benchmarks</a></li>
        <li><a href="{up}blog/" aria-current="page">Writing</a></li>
        <li><a href="{up}#about">About</a></li>
        <li><a href="https://github.com/paperniuk" rel="me">GitHub</a></li>
      </ul>
    </nav>
  </header>
"""


def foot(depth):
    up = "../" * depth
    return f"""  <footer class="site-footer">
    <div class="container footer-grid">
      <div>
        <p class="footer-name">Erik Paperniuk</p>
        <p class="footer-title">System Integration &amp; Automation Engineer</p>
      </div>
      <ul class="footer-links">
        <li><a href="{up}blog/">Writing</a></li>
        <li><a href="https://github.com/paperniuk" rel="me">GitHub</a></li>
        <li><a href="https://www.linkedin.com/in/paperniuk" rel="me">LinkedIn</a></li>
        <li><a href="https://www.reddit.com/user/Erp4759/" rel="me">u/Erp4759 on Reddit</a></li>
        <li><a href="{up}blog/feed.xml">Feed</a></li>
      </ul>
    </div>
  </footer>
</body>
</html>
"""


def pretty(when):
    return f"{when.day} {when.strftime('%b %Y')}"


def post_page(meta):
    url = f"{SITE}/blog/{meta['slug']}/"
    first_img = re.search(r"\]\((img/[^)]+)\)", meta["body"])
    image = f"{SITE}/blog/{first_img.group(1)}" if first_img else f"{SITE}/img/erik.jpg"
    ld = (
        '  <script type="application/ld+json">{"@context":"https://schema.org","@type":"BlogPosting",'
        f'"headline":{html_json(meta["title"])},"datePublished":"{meta["date"]}",'
        f'"description":{html_json(meta["summary"])},"image":"{image}","url":"{url}",'
        '"author":{"@type":"Person","name":"Erik Paperniuk","url":"https://paperniuk.github.io/"}}</script>\n'
    )
    extra = (
        '  <meta property="og:type" content="article">\n'
        f'  <meta property="og:image" content="{image}">\n'
        '  <meta name="twitter:card" content="summary_large_image">\n' + ld
    )
    tags = "".join(f'<li class="tag">{esc(t)}</li>' for t in meta["tags"])
    reddit = (
        f'<a class="link-arrow" href="{esc(meta["reddit"])}">Discussion on Reddit {ARROW_OUT}</a>'
        if meta.get("reddit")
        else ""
    )
    return (
        head(f"{meta['title']} · Erik Paperniuk", meta["summary"], url, 2, extra)
        + f"""  <main id="main-content" class="post-wrap">
    <div class="container">
      <header class="post-header">
        <a class="back" href="../">&larr; all posts</a>
        <h1>{esc(meta['title'])}</h1>
        <p class="lede">{esc(meta['summary'])}</p>
        <p class="post-meta" style="margin-top:1.2rem"><time datetime="{meta['date']}">{pretty(meta['when'])}</time><span>{meta['minutes']} min read</span></p>
        <ul class="tags" style="margin-top:0.8rem" aria-label="Tags">{tags}</ul>
      </header>
      <article class="post">
{render_body(meta)}
      </article>
      <div class="post-foot">
        <a class="link-arrow" href="https://github.com/paperniuk">GitHub {ARROW_OUT}</a>
        {reddit}
        <a class="link-arrow" href="../">All posts {ARROW_IN}</a>
      </div>
    </div>
  </main>
"""
        + foot(2)
    )


def html_json(text):
    import json
    return json.dumps(text, ensure_ascii=False).replace("</", "<\\/")


def index_page(posts):
    items = "".join(
        f"""        <li class="post-item"><a href="{p['slug']}/">
          <p class="post-meta"><time datetime="{p['date']}">{pretty(p['when'])}</time><span>{p['minutes']} min read</span><span>{esc(', '.join(p['tags']))}</span></p>
          <h2>{esc(p['title'])}</h2>
          <p>{esc(p['summary'])}</p>
        </a></li>
"""
        for p in posts
    )
    return (
        head("Writing · Erik Paperniuk", "Write-ups of Metal kernel and local inference work on Apple Silicon, with benchmarks.", f"{SITE}/blog/", 1,
             '  <meta property="og:type" content="website">\n  <meta name="twitter:card" content="summary">\n')
        + f"""  <main id="main-content">
    <div class="blog-head">
      <div class="container">
        <p class="eyebrow">Writing</p>
        <h1 class="hero-name">Notes with the numbers</h1>
        <p class="hero-lede" style="margin-bottom:0">Write-ups of the kernel and inference work, with the charts and the commands to reproduce them. Most of them started as posts on r/LocalLLaMA. <a class="link-arrow" href="feed.xml">Feed</a></p>
      </div>
    </div>
    <div class="container" style="padding-block:1rem 4rem">
      <ul class="post-list">
{items}      </ul>
    </div>
  </main>
"""
        + foot(1)
    )


def feed(posts):
    updated = max(p["when"] for p in posts).strftime("%Y-%m-%dT00:00:00Z")
    entries = "".join(
        f"""  <entry>
    <title>{esc(p['title'])}</title>
    <link href="{SITE}/blog/{p['slug']}/"/>
    <id>{SITE}/blog/{p['slug']}/</id>
    <updated>{p['when'].strftime('%Y-%m-%dT00:00:00Z')}</updated>
    <summary>{esc(p['summary'])}</summary>
  </entry>
"""
        for p in posts
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Erik Paperniuk, notes</title>
  <link href="{SITE}/blog/feed.xml" rel="self"/>
  <link href="{SITE}/blog/"/>
  <id>{SITE}/blog/</id>
  <updated>{updated}</updated>
  <author><name>Erik Paperniuk</name></author>
{entries}</feed>
"""


def sitemap(posts):
    urls = [(f"{SITE}/", None), (f"{SITE}/research/metafake/", None), (f"{SITE}/blog/", None)] + [
        (f"{SITE}/blog/{p['slug']}/", p["date"]) for p in posts
    ]
    rows = "".join(
        f"  <url><loc>{u}</loc>" + (f"<lastmod>{d}</lastmod>" if d else "") + "</url>\n" for u, d in urls
    )
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{rows}</urlset>\n'


def home_rows(posts):
    return "\n".join(
        f"""            <li class="work-row reveal">
              <a class="work-link" href="blog/{p['slug']}/">
                <span class="work-name"><span class="work-date">{pretty(p['when'])} · {p['minutes']} min</span>{esc(p['title'])}</span>
                <span class="work-tagline">{esc(p['summary'])}</span>
                {ARROW_OUT.replace('class="icon"', 'class="icon work-arrow"')}
              </a>
            </li>"""
        for p in posts[:HOME_POSTS]
    )


def main():
    posts = sorted(
        (parse_post(p) for p in (ROOT / "posts").glob("*.md")), key=lambda m: m["when"], reverse=True
    )
    if not posts:
        sys.exit("no posts found")
    blog = ROOT / "blog"
    for meta in posts:
        out = blog / meta["slug"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "index.html").write_text(post_page(meta), encoding="utf-8")
    (blog / "index.html").write_text(index_page(posts), encoding="utf-8")
    (blog / "feed.xml").write_text(feed(posts), encoding="utf-8")
    (ROOT / "sitemap.xml").write_text(sitemap(posts), encoding="utf-8")

    home = ROOT / "index.html"
    text = home.read_text(encoding="utf-8")
    text, n = re.subn(
        r"(<!-- posts:start -->\n).*?(<!-- posts:end -->)",
        lambda m: m.group(1) + home_rows(posts) + "\n" + m.group(2),
        text,
        flags=re.S,
    )
    if n != 1:
        sys.exit("index.html: posts markers not found")
    home.write_text(text, encoding="utf-8")
    print(f"built {len(posts)} posts")


if __name__ == "__main__":
    main()
