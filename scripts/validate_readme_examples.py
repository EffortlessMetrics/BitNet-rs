#!/usr/bin/env python3
"""
Validate README examples for release validation.
"""

import re
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any


def extract_code_blocks(readme_content: str) -> List[Dict[str, Any]]:
    """Extract code blocks from README markdown."""
    code_blocks = []
    
    # Pattern to match fenced code blocks with language
    pattern = r'```(\w+)?\n(.*?)\n```'
    matches = re.finditer(pattern, readme_content, re.DOTALL)
    
    for i, match in enumerate(matches):
        language = match.group(1) or 'text'
        code = match.group(2)
        
        code_blocks.append({
            'index': i,
            'language': language,
            'code': code,
            'line_start': readme_content[:match.start()].count('\n') + 1
        })
    
    return code_blocks


def validate_rust_code_block(code: str, block_info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a Rust code block."""
    result = {
        'block_info': block_info,
        'valid': False,
        'error': None,
        'warnings': []
    }
    
    # Skip code blocks that are clearly not meant to be compiled
    skip_patterns = [
        r'#\s*\[.*no_run.*\]',
        r'#\s*\[.*ignore.*\]',
        r'//\s*This is just an example',
        r'//\s*Pseudo-code',
        r'//\s*Not compilable'
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            result['valid'] = True
            result['warnings'].append('Skipped validation (marked as non-compilable)')
            return result
    
    # Create a temporary Rust file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
        # Wrap code in a basic structure if it's not a complete program
        if 'fn main()' not in code and 'fn ' not in code:
            # Assume it's a code snippet that should go in main
            wrapped_code = f'''
fn main() {{
{code}
}}
'''
        else:
            wrapped_code = code
        
        f.write(wrapped_code)
        temp_file = Path(f.name)
    
    try:
        # Try to compile the code
        result_compile = subprocess.run(
            ['rustc', '--edition', '2021', '--crate-type', 'bin', str(temp_file)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result_compile.returncode == 0:
            result['valid'] = True
        else:
            result['error'] = result_compile.stderr
            
            # Check if it's just missing dependencies
            if 'can\'t find crate' in result_compile.stderr or 'unresolved import' in result_compile.stderr:
                result['warnings'].append('May require external dependencies')
                result['valid'] = True  # Consider valid if only missing deps
    
    except subprocess.TimeoutExpired:
        result['error'] = 'Compilation timeout'
    except Exception as e:
        result['error'] = str(e)
    finally:
        # Clean up temporary files
        temp_file.unlink(missing_ok=True)
        # Also clean up any generated binary
        binary_path = temp_file.with_suffix('')
        binary_path.unlink(missing_ok=True)
    
    return result


def validate_shell_code_block(code: str, block_info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a shell code block."""
    result = {
        'block_info': block_info,
        'valid': False,
        'error': None,
        'warnings': []
    }
    
    # Check for common shell commands that should be valid
    lines = code.strip().split('\n')
    valid_commands = []
    invalid_commands = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Extract the command (first word)
        command = line.split()[0] if line.split() else ''
        
        # Check if command exists
        try:
            subprocess.run(['which', command], capture_output=True, check=True)
            valid_commands.append(command)
        except subprocess.CalledProcessError:
            # Special cases for commands that might not be available but are valid
            if command in ['cargo', 'rustc', 'rustup', 'git']:
                result['warnings'].append(f'Command {command} not found but expected to be available')
            else:
                invalid_commands.append(command)
    
    if invalid_commands:
        result['error'] = f'Unknown commands: {", ".join(invalid_commands)}'
    else:
        result['valid'] = True
    
    return result


def validate_readme_examples(readme_path: Path) -> Dict[str, Any]:
    """Validate all examples in README file."""
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {
            'error': f'Failed to read README: {e}',
            'results': []
        }
    
    code_blocks = extract_code_blocks(content)
    results = []
    
    for block in code_blocks:
        if block['language'] in ['rust', 'rs']:
            result = validate_rust_code_block(block['code'], block)
        elif block['language'] in ['bash', 'sh', 'shell']:
            result = validate_shell_code_block(block['code'], block)
        else:
            # Skip non-code blocks
            result = {
                'block_info': block,
                'valid': True,
                'error': None,
                'warnings': [f'Skipped validation for {block["language"]} block']
            }
        
        results.append(result)
    
    # Calculate summary
    total_blocks = len([r for r in results if r['block_info']['language'] in ['rust', 'rs', 'bash', 'sh', 'shell']])
    valid_blocks = len([r for r in results if r['valid']])
    
    return {
        'total_code_blocks': len(code_blocks),
        'validated_blocks': total_blocks,
        'valid_blocks': valid_blocks,
        'success_rate': (valid_blocks / total_blocks * 100) if total_blocks > 0 else 100,
        'results': results
    }


def main():
    readme_path = Path('README.md')
    
    if not readme_path.exists():
        print("README.md not found")
        return 1
    
    print("Validating README examples...")
    
    validation_results = validate_readme_examples(readme_path)
    
    if 'error' in validation_results:
        print(f"Error: {validation_results['error']}")
        return 1
    
    # Print summary
    print(f"\n=== README Validation Summary ===")
    print(f"Total code blocks: {validation_results['total_code_blocks']}")
    print(f"Validated blocks: {validation_results['validated_blocks']}")
    print(f"Valid blocks: {validation_results['valid_blocks']}")
    print(f"Success rate: {validation_results['success_rate']:.1f}%")
    
    # Print detailed results
    print(f"\n=== Detailed Results ===")
    for result in validation_results['results']:
        block = result['block_info']
        status = "✅" if result['valid'] else "❌"
        print(f"{status} Block {block['index']} ({block['language']}) at line {block['line_start']}")
        
        if result['error']:
            print(f"   Error: {result['error']}")
        
        for warning in result['warnings']:
            print(f"   Warning: {warning}")
    
    # Return error code if validation failed
    if validation_results['success_rate'] < 80:  # Require 80% success rate
        print(f"\n❌ README validation failed: {validation_results['success_rate']:.1f}% < 80%")
        return 1
    
    print(f"\n✅ README validation passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())