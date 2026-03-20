#!/usr/bin/env python3
"""Fetch AI news from RSS/Atom feeds + HuggingFace trending GGUF models, write feed.json."""

import json
import re
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET
import requests

RSS_FEEDS = [
    ("https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "The Verge"),
    ("https://techcrunch.com/category/artificial-intelligence/feed/", "TechCrunch"),
]

HF_API = "https://huggingface.co/api/models"
HEADERS = {"User-Agent": "MyAI-FeedBot/1.0 (https://github.com/BittieDeluxe/myai_catalog)"}


def strip_namespaces(xml_bytes: bytes) -> bytes:
    """Strip XML namespace declarations and prefixes so ElementTree tags are plain."""
    s = xml_bytes.decode('utf-8', errors='replace')
    s = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', '', s)       # remove xmlns attrs
    s = re.sub(r'<(/?)\w+:(\w+)', r'<\1\2', s)             # strip ns prefixes from tags
    s = re.sub(r'\s\w+:(\w+)=', r' \1=', s)                # strip ns prefixes from attrs
    return s.encode('utf-8')


def strip_html(html: str) -> str:
    text = re.sub(r'<[^>]+>', '', html)
    return (text
            .replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            .replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
            .strip())


def to_iso8601(date_str: str) -> str | None:
    if not date_str:
        return None
    # Try RFC 822 (RSS)
    try:
        dt = parsedate_to_datetime(date_str.strip())
        return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        pass
    # Try ISO 8601 (Atom)
    for fmt in ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S%z'):
        try:
            dt = datetime.strptime(date_str.strip()[:25], fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception:
            pass
    return None


def fetch_rss(url: str, source: str) -> list[dict]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"Warning: could not fetch {source}: {e}", file=sys.stderr)
        return []

    try:
        root = ET.fromstring(strip_namespaces(r.content))
    except ET.ParseError as e:
        print(f"Warning: could not parse {source} XML: {e}", file=sys.stderr)
        return []

    # Support both RSS <item> and Atom <entry>
    entries = root.findall('.//item') or root.findall('.//entry')

    items = []
    for entry in entries:
        title = (entry.findtext('title') or '').strip()
        if not title:
            continue

        # RSS: <link> text node. Atom: <link href="..." rel="alternate"/>
        link = ''
        link_el = entry.find('link')
        if link_el is not None:
            link = link_el.get('href') or (link_el.text or '')
        link = link.strip()
        if not link:
            continue

        # Date: RSS uses pubDate, Atom uses published or updated
        pub = (entry.findtext('pubDate') or
               entry.findtext('published') or
               entry.findtext('updated') or '')

        # Description: RSS uses description, Atom uses summary or content
        raw_desc = (entry.findtext('description') or
                    entry.findtext('summary') or
                    entry.findtext('content') or '')
        snippet = strip_html(raw_desc)[:200] or None

        items.append({
            'title':       title,
            'url':         link,
            'source':      source,
            'publishedAt': to_iso8601(pub),
            'snippet':     snippet,
        })

    return items[:15]


def fetch_trending() -> list[dict]:
    try:
        r = requests.get(HF_API, headers=HEADERS, timeout=10, params={
            'sort': 'trendingScore', 'direction': '-1', 'limit': '10', 'filter': 'gguf',
        })
        r.raise_for_status()
        models = r.json()
    except Exception as e:
        print(f"Warning: could not fetch HF trending: {e}", file=sys.stderr)
        return []

    results = []
    for m in models:
        model_id  = m.get('id', '')
        name_part = model_id.split('/')[-1] if '/' in model_id else model_id
        for token in ['-GGUF', '-gguf', '-Instruct', '-instruct', '-it', '-chat', '-Chat']:
            name_part = name_part.replace(token, '')
        name_part = re.sub(r'-[vV]\d+\.\d+$', '', name_part).replace('-', ' ').strip()
        results.append({
            'id':          model_id,
            'displayName': name_part,
            'downloads':   m.get('downloads', 0),
        })

    return results


def main():
    news, seen = [], set()
    for url, source in RSS_FEEDS:
        for item in fetch_rss(url, source):
            if item['url'] not in seen:
                seen.add(item['url'])
                news.append(item)

    news.sort(key=lambda x: x['publishedAt'] or '', reverse=True)
    trending = fetch_trending()

    feed = {
        'updated':  datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'news':     news,
        'trending': trending,
    }

    with open('feed.json', 'w') as f:
        json.dump(feed, f, indent=2)

    print(f"Done: {len(news)} news items, {len(trending)} trending models")


if __name__ == '__main__':
    main()
