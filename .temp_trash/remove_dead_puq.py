import sys, os
sys.path.insert(0, os.path.abspath('.'))

path = 'ai_chatbot.py'
lines = open(path, encoding='utf-8').read().split('\n')

start, end = 531, 648  # 1-based inclusive start, exclusive end

# Safety assertions
assert lines[start-1].strip().startswith('async def process_unbiased_query'), lines[start-1]
assert lines[end-1].strip().startswith('def _generate_opportunity_charts'), lines[end-1]

removed = lines[start-1:end-1]
del lines[start-1:end-1]
open(path, 'w', encoding='utf-8').write('\n'.join(lines))
print(f'REMOVED dead process_unbiased_query ({len(removed)} lines)')
print('FIRST LINE REMOVED:', removed[0].strip()[:80])
print('LAST LINE REMOVED:', removed[-1].strip()[:80])
