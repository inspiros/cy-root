#!/usr/bin/env python3
"""Keep docs canonical and adapt repository-only links for the rendered manual."""
import json
import re
import sys

REPOSITORY = 'https://github.com/inspiros/cy-root/blob/v2.0.0/'


def destination(target):
    if target == '../README.md':
        return REPOSITORY + 'README.md'
    if target.startswith('../assets/'):
        return 'assets/' + target.removeprefix('../assets/')
    if target.startswith('../'):
        return REPOSITORY + target.removeprefix('../')
    if target == 'README.md' or target.startswith('README.md#'):
        return target.replace('README.md', 'index.md', 1)
    return target


def preserve_math(source):
    # Markdown otherwise consumes TeX delimiters and escapes such as \|.
    # Entities become ordinary TeX text nodes in the browser, for MathJax to
    # typeset. Fenced and inline code remain literal examples.
    pieces = re.split(r'(```.*?```|`[^`]*`)', source, flags=re.S)
    def replacement(match):
        tex = match.group(0)
        kind = 'math-display' if tex.startswith(r'\[') else 'math-inline'
        tag = 'div' if kind == 'math-display' else 'span'
        encoded = ''.join(f'&#{ord(char)};' for char in tex)
        return f'<{tag} class="{kind}">{encoded}</{tag}>'
    for index in range(0, len(pieces), 2):
        pieces[index] = re.sub(r'\\\((.*?)\\\)|\\\[(.*?)\\\]', replacement,
                               pieces[index], flags=re.S)
    return ''.join(pieces)


def chapters(sections):
    for section in sections:
        chapter = section.get('Chapter')
        if chapter is None:
            continue
        content = chapter['content']
        # Includes are expanded by mdBook's links preprocessor first. Relative
        # documentation links stay local; source/example links go to the release.
        content = re.sub(r'(?P<start>\]\()(?P<url>[^\s)]+)',
                         lambda m: m['start'] + destination(m['url']), content)
        content = re.sub(r'(?P<start>(?:href|src)=["\'])(?P<url>[^"\']+)',
                         lambda m: m['start'] + destination(m['url']), content)
        # mdBook's HTML renderer rewrites Markdown .md links. Raw HTML cards
        # need the same extension mapping explicitly.
        content = re.sub(r'(?P<start>href=["\'])(?P<path>[^"\':#]+)\.md(?P<end>["\'])',
                         lambda m: m['start'] + m['path'] + '.html' + m['end'], content)
        chapter['content'] = preserve_math(content)
        chapters(chapter.get('sub_items', []))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'supports':
        sys.exit(0 if sys.argv[-1] == 'html' else 1)
    context, book = json.load(sys.stdin)
    chapters(book['sections'] if 'sections' in book else book['items'])
    json.dump(book, sys.stdout)
