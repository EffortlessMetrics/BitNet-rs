#!/usr/bin/env python3
"""
Fix markdownlint issues in documentation files.
Applies mechanical fixes following markdownlint rules.
"""

import re
import sys

def fix_markdown(content):
    """Apply all markdownlint fixes to content."""

    # Remove emojis from headings (MD026)
    content = re.sub(r'(###?\s+.*?):\s*[✅⚠❌️]', r'\1', content)
    content = re.sub(r'(###?\s+.*?)[✅⚠❌️]\s*', r'\1', content)

    # Remove emojis from inline bold text
    content = re.sub(r'\*\*([^*]+?)[✅⚠❌️]\*\*', r'**\1**', content)

    # Fix "Status:" patterns with emojis
    content = re.sub(r'\*\*Status\*\*:\s*[✅⚠❌️]\s*\*\*([^*]+?)\*\*', r'**Status**: \1', content)

    # Fix lines with just emoji bullets
    content = re.sub(r'^([✅⚠❌])\s+\*\*', r'- **', content, flags=re.MULTILINE)

    # Fix "Gap X:" patterns
    content = re.sub(r'\*\*Gap \d+:\s*([^*]+?)\*\*', r'**Gap: \1**', content)

    # Ensure blank line after headings (MD022)
    content = re.sub(r'(^#{1,6}\s+.+$)\n([^#\n-])', r'\1\n\n\2', content, flags=re.MULTILINE)

    # Ensure blank line before headings (MD022)
    content = re.sub(r'([^\n])\n(^#{1,6}\s+)', r'\1\n\n\2', content, flags=re.MULTILINE)

    # Ensure blank lines around code blocks (MD031)
    content = re.sub(r'([^\n])\n(```)', r'\1\n\n\2', content, flags=re.MULTILINE)
    content = re.sub(r'(```)\n([^\n])', r'\1\n\n\2', content, flags=re.MULTILINE)

    # Ensure blank lines around lists (MD032)
    content = re.sub(r'([^\n])\n(^[-*+]\s)', r'\1\n\n\2', content, flags=re.MULTILINE)
    content = re.sub(r'(^[-*+]\s.+$)\n([^-*+\n])', r'\1\n\n\2', content, flags=re.MULTILINE)

    return content

def main():
    files = [
        'ci/DOCUMENTATION_NAVIGATION_ASSESSMENT.md',
        'AGENT_ORCHESTRATION_SUMMARY.md'
    ]

    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            fixed_content = fix_markdown(content)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            print(f"✓ Fixed {filepath}")
        except Exception as e:
            print(f"✗ Error fixing {filepath}: {e}", file=sys.stderr)
            return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
