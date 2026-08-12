import sys, os
sys.path.insert(0, os.path.abspath('.'))

path = 'ai_chatbot.py'
lines = open(path, encoding='utf-8').read().split('\n')

FIRST_OCCURRENCES = [2333, 2410, 2774]  # 1-based line of each dead first def

def method_end(start_line, lines):
    """Return index (0-based, exclusive) of the line after the method body."""
    i = start_line - 1
    indent = len(lines[i]) - len(lines[i].lstrip())
    j = i + 1
    while j < len(lines):
        l = lines[j]
        if l.strip() == '':
            j += 1
            continue
        this_indent = len(l) - len(l.lstrip())
        if this_indent <= indent:
            break
        j += 1
    return j

# Sort descending so line numbers stay valid as we delete.
for start in sorted(FIRST_OCCURRENCES, reverse=True):
    end = method_end(start, lines)
    removed = lines[start-1:end]
    # Safety: confirm the removed block is a def we expect and is NOT the last
    # definition of its name.
    def_line = removed[0].strip() if removed else ''
    assert def_line.startswith('def '), f'not a def at {start}: {def_line!r}'
    name = def_line.split('(')[0].split('def ')[1].strip()
    # Verify another definition of the same name exists later in the file.
    later = [i for i in range(end, len(lines)) if lines[i].strip().startswith(f'def {name}')]
    assert later, f'no later definition of {name} — refusing to delete the only one'
    print(f'REMOVING dead {name} @ {start} ({end-start} lines)')
    del lines[start-1:end]

open(path, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE')
