"""
Script to remove all emojis from Python files in the codebase.
"""

import os
import re
from pathlib import Path

# Comprehensive emoji pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "]+", flags=re.UNICODE
)

# Common emojis to remove (as fallback)
COMMON_EMOJIS = [
    '📊', '📈', '📉', '🎯', '💡', '⚠️', '✅', '❌', '🔍', '🚀',
    '💰', '📄', '⏱️', '🛡️', '🟢', '🔴', '⚪', '🟡', '▶️', '⏸️',
    '⏹️', '🔄', '➕', '📤', '📝', '📋', '📏', '🧠', '⚙️', '💾',
    '🎭', '💓', '😊', '👍', '❤️', '🙌', '😂', '💀', '🤔', '👀',
    '➡️', '👆', '🔥', '💵', '📱', '💻', '🌐', '🔔', '⭐', '✨'
]

def remove_emojis_from_text(text):
    """Remove all emojis from text."""
    # Remove using regex pattern
    text = EMOJI_PATTERN.sub('', text)
    
    # Remove common emojis as fallback
    for emoji in COMMON_EMOJIS:
        text = text.replace(emoji, '')
    
    return text

def process_file(filepath):
    """Process a single Python file to remove emojis."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        cleaned_content = remove_emojis_from_text(content)
        
        if cleaned_content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            return True, filepath
        return False, None
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False, None

def main():
    """Remove emojis from all Python files in the current directory."""
    current_dir = Path('.')
    python_files = list(current_dir.glob('*.py'))
    
    modified_files = []
    
    print(f"Found {len(python_files)} Python files to process...")
    
    for filepath in python_files:
        if filepath.name == 'remove_emojis.py':
            continue  # Skip this script itself
        
        modified, path = process_file(filepath)
        if modified:
            modified_files.append(path)
            print(f"✓ Cleaned: {path}")
    
    print(f"\nCompleted! Modified {len(modified_files)} files.")
    if modified_files:
        print("\nModified files:")
        for f in modified_files:
            print(f"  - {f}")

if __name__ == "__main__":
    main()
