#!/usr/bin/env python3
"""Check local documentation links and optionally execute its Python examples."""
from __future__ import annotations

import argparse
import contextlib
from html import unescape
import io
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
FENCES = re.compile(r'^```[^\n]*\n.*?^```[^\n]*$', re.M | re.S)
PYTHON = re.compile(r'^```python[^\n]*\n(.*?)^```', re.M | re.S)


def anchor_ids(path):
    source = FENCES.sub('', path.read_text())
    anchors = set(re.findall(r'\bid=["\']([^"\']+)', source))
    counts = {}
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', source, re.M):
        heading = re.sub(r'<[^>]+>', '', heading)
        heading = re.sub(r'\[([^]]+)\]\([^)]*\)', r'\1', heading)
        slug = re.sub(r'[^\w\-\s]', '', unescape(heading).lower())
        slug = re.sub(r'\s', '-', slug)
        index = counts.get(slug, 0)
        counts[slug] = index + 1
        anchors.add(slug if index == 0 else f'{slug}-{index}')
    return anchors


def relative_links(path):
    source = FENCES.sub('', path.read_text())
    # Inline code can contain illustrative paths that are not links.
    source = re.sub(r'`[^`]+`', '', source)
    candidates = re.findall(r'\]\(\s*(<[^>]+>|[^\s)]+)', source)
    candidates += re.findall(r'^\s*\[[^]]+\]:\s*(<[^>]+>|\S+)', source, re.M)
    candidates += re.findall(r'\b(?:src|srcset|href)=["\']([^"\']+)', source)
    for raw in candidates:
        target = unescape(raw.strip('<>'))
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not target:
            continue
        yield target, parsed


def check_links(paths):
    checked, errors = 0, []
    for path in paths:
        for target, parsed in relative_links(path):
            checked += 1
            dest = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if not dest.exists():
                errors.append(f'{path.relative_to(ROOT)}: missing {target}')
            elif parsed.fragment and dest.suffix.lower() == '.md':
                if unquote(parsed.fragment) not in anchor_ids(dest):
                    errors.append(f'{path.relative_to(ROOT)}: missing anchor {target}')
    return checked, errors


def check_snippets(paths, compiled_example_dir):
    if compiled_example_dir:
        sys.path.insert(0, str(compiled_example_dir.resolve()))
    checked, errors, records = 0, [], []
    for path in paths:
        namespace = {'__name__': '__doc_example__'}
        for index, match in enumerate(PYTHON.finditer(path.read_text()), 1):
            checked += 1
            label = f'{path.relative_to(ROOT)}: Python block {index}'
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    exec(compile(match.group(1), label, 'exec'), namespace)
                records.append({'example': label, 'status': 'ok', 'output': output.getvalue()})
            except Exception as error:
                message = f'{label}: {type(error).__name__}: {error}'
                errors.append(message)
                records.append({'example': label, 'status': 'error', 'error': str(error)})
    return checked, errors, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snippets', action='store_true')
    parser.add_argument('--compiled-example-dir', type=Path)
    parser.add_argument('--report', type=Path, help='optional JSON evidence outside the source tree')
    args = parser.parse_args()
    paths = [ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md')),
             *sorted((ROOT / 'examples').glob('*.md'))]
    links, errors = check_links(paths)
    print(f'Relative links and Markdown anchors: {links} checked, {len(errors)} errors')
    records, snippets = [], 0
    if args.snippets:
        snippets, failures, records = check_snippets(paths, args.compiled_example_dir)
        errors.extend(failures)
        print(f'Python snippets: {snippets} checked, {len(failures)} errors')
    for error in errors:
        print(error, file=sys.stderr)
    if args.report:
        args.report.write_text(json.dumps({'links_checked': links, 'snippets_checked': snippets,
                                          'errors': errors, 'snippets': records}, indent=2) + '\n')
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
