#!/usr/bin/env python3
"""
Check documentation links for release validation.
"""

import re
import sys
import requests
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse
import time


def find_markdown_files(root_dir: Path) -> List[Path]:
    """Find all markdown files in the project."""
    markdown_files = []

    # Search in docs/, README files, and other common locations
    patterns = [
        "*.md",
        "docs/**/*.md",
        "**/*.md"
    ]

    for pattern in patterns:
        markdown_files.extend(root_dir.glob(pattern))

    # Filter out target directory and node_modules
    filtered_files = []
    for file_path in markdown_files:
        path_str = str(file_path)
        if not any(exclude in path_str for exclude in ['target/', 'node_modules/', '.git/']):
            filtered_files.append(file_path)

    return list(set(filtered_files))  # Remove duplicates


def extract_links(content: str, file_path: Path) -> List[Dict[str, Any]]:
    """Extract all links from markdown content."""
    links = []

    # Pattern for markdown links: [text](url)
    markdown_link_pattern = r'\[([^\]]*)\]\(([^)]+)\)'

    # Pattern for reference links: [text][ref] and [ref]: url
    reference_pattern = r'\[([^\]]+)\]:\s*(.+)'
    reference_usage_pattern = r'\[([^\]]*)\]\[([^\]]+)\]'

    # Find all markdown links
    for match in re.finditer(markdown_link_pattern, content):
        text = match.group(1)
        url = match.group(2)
        line_num = content[:match.start()].count('\n') + 1

        links.append({
            'text': text,
            'url': url,
            'line': line_num,
            'type': 'direct',
            'file': file_path
        })

    # Find reference definitions
    references = {}
    for match in re.finditer(reference_pattern, content):
        ref_id = match.group(1).lower()
        url = match.group(2)
        references[ref_id] = url

    # Find reference usages
    for match in re.finditer(reference_usage_pattern, content):
        text = match.group(1)
        ref_id = match.group(2).lower()
        line_num = content[:match.start()].count('\n') + 1

        if ref_id in references:
            links.append({
                'text': text,
                'url': references[ref_id],
                'line': line_num,
                'type': 'reference',
                'file': file_path
            })

    return links


def categorize_link(url: str, file_path: Path) -> str:
    """Categorize a link as internal, external, or anchor."""
    if url.startswith('#'):
        return 'anchor'
    elif url.startswith('http://') or url.startswith('https://'):
        return 'external'
    elif url.startswith('mailto:'):
        return 'email'
    else:
        return 'internal'


def check_internal_link(url: str, file_path: Path, root_dir: Path) -> Dict[str, Any]:
    """Check if an internal link is valid."""
    result = {
        'valid': False,
        'error': None,
        'resolved_path': None
    }

    try:
        # Handle relative paths
        if url.startswith('./') or url.startswith('../') or not url.startswith('/'):
            # Relative to current file
            target_path = (file_path.parent / url).resolve()
        else:
            # Absolute path from root
            target_path = (root_dir / url.lstrip('/')).resolve()

        # Check if target exists
        if target_path.exists():
            result['valid'] = True
            result['resolved_path'] = str(target_path)
        else:
            result['error'] = f'File not found: {target_path}'

    except Exception as e:
        result['error'] = str(e)

    return result


def check_external_link(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Check if an external link is accessible."""
    result = {
        'valid': False,
        'status_code': None,
        'error': None
    }

    try:
        # Add user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; LinkChecker/1.0)'
        }

        response = requests.head(url, timeout=timeout, headers=headers, allow_redirects=True)
        result['status_code'] = response.status_code

        if response.status_code < 400:
            result['valid'] = True
        else:
            result['error'] = f'HTTP {response.status_code}'

    except requests.exceptions.Timeout:
        result['error'] = 'Timeout'
    except requests.exceptions.ConnectionError:
        result['error'] = 'Connection error'
    except requests.exceptions.RequestException as e:
        result['error'] = str(e)
    except Exception as e:
        result['error'] = str(e)

    return result


def check_anchor_link(anchor: str, content: str) -> Dict[str, Any]:
    """Check if an anchor link exists in the content."""
    result = {
        'valid': False,
        'error': None
    }

    # Remove the # prefix
    anchor_id = anchor.lstrip('#')

    # Look for headers that would generate this anchor
    # GitHub-style anchor generation: lowercase, replace spaces with hyphens, remove special chars
    header_patterns = [
        rf'^#+\s+.*{re.escape(anchor_id)}.*$',  # Direct match
        rf'^#+\s+.*{re.escape(anchor_id.replace("-", " "))}.*$',  # With spaces
    ]

    for pattern in header_patterns:
        if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
            result['valid'] = True
            break

    if not result['valid']:
        result['error'] = f'Anchor not found: {anchor_id}'

    return result


def validate_links_in_file(file_path: Path, root_dir: Path) -> Dict[str, Any]:
    """Validate all links in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {
            'file': str(file_path),
            'error': f'Failed to read file: {e}',
            'links': []
        }

    links = extract_links(content, file_path)
    results = []

    for link in links:
        url = link['url']
        category = categorize_link(url, file_path)

        link_result = {
            'link': link,
            'category': category,
            'valid': False,
            'error': None,
            'details': {}
        }

        if category == 'internal':
            check_result = check_internal_link(url, file_path, root_dir)
            link_result.update(check_result)
            link_result['details'] = check_result
        elif category == 'external':
            check_result = check_external_link(url)
            link_result.update(check_result)
            link_result['details'] = check_result
        elif category == 'anchor':
            check_result = check_anchor_link(url, content)
            link_result.update(check_result)
            link_result['details'] = check_result
        elif category == 'email':
            # Email links are generally valid if they have @ symbol
            link_result['valid'] = '@' in url
            if not link_result['valid']:
                link_result['error'] = 'Invalid email format'

        results.append(link_result)

    return {
        'file': str(file_path.relative_to(root_dir)),
        'total_links': len(links),
        'results': results
    }


def validate_all_links(root_dir: Path, max_workers: int = 10) -> Dict[str, Any]:
    """Validate links in all markdown files."""
    markdown_files = find_markdown_files(root_dir)

    print(f"Found {len(markdown_files)} markdown files to check")

    all_results = []

    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(validate_links_in_file, file_path, root_dir): file_path
            for file_path in markdown_files
        }

        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                all_results.append(result)
                print(f"Checked {result['file']}: {result['total_links']} links")
            except Exception as e:
                print(f"Error checking {file_path}: {e}")

    # Calculate summary statistics
    total_files = len(all_results)
    total_links = sum(r['total_links'] for r in all_results)

    all_link_results = []
    for file_result in all_results:
        all_link_results.extend(file_result['results'])

    valid_links = len([r for r in all_link_results if r['valid']])
    invalid_links = len([r for r in all_link_results if not r['valid']])

    # Categorize by type
    by_category = {}
    for link_result in all_link_results:
        category = link_result['category']
        if category not in by_category:
            by_category[category] = {'total': 0, 'valid': 0, 'invalid': 0}

        by_category[category]['total'] += 1
        if link_result['valid']:
            by_category[category]['valid'] += 1
        else:
            by_category[category]['invalid'] += 1

    return {
        'summary': {
            'total_files': total_files,
            'total_links': total_links,
            'valid_links': valid_links,
            'invalid_links': invalid_links,
            'success_rate': (valid_links / total_links * 100) if total_links > 0 else 100
        },
        'by_category': by_category,
        'file_results': all_results,
        'invalid_links': [r for r in all_link_results if not r['valid']]
    }


def main():
    root_dir = Path.cwd()

    print("Checking documentation links...")

    results = validate_all_links(root_dir)

    # Print summary
    summary = results['summary']
    print(f"\n=== Link Validation Summary ===")
    print(f"Files checked: {summary['total_files']}")
    print(f"Total links: {summary['total_links']}")
    print(f"Valid links: {summary['valid_links']}")
    print(f"Invalid links: {summary['invalid_links']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")

    # Print by category
    print(f"\n=== By Category ===")
    for category, stats in results['by_category'].items():
        print(f"{category}: {stats['valid']}/{stats['total']} valid ({stats['valid']/stats['total']*100:.1f}%)")

    # Print invalid links
    if results['invalid_links']:
        print(f"\n=== Invalid Links ===")
        for link_result in results['invalid_links'][:20]:  # Limit to first 20
            link = link_result['link']
            print(f"❌ {link['file']}:{link['line']} - {link['url']}")
            if link_result['error']:
                print(f"   Error: {link_result['error']}")

    # Return error code if too many links are broken
    if summary['success_rate'] < 90:  # Require 90% success rate
        print(f"\n❌ Link validation failed: {summary['success_rate']:.1f}% < 90%")
        return 1

    print(f"\n✅ Link validation passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
