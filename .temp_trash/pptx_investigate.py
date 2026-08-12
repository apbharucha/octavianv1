import re

src = open('presentation_generator.py', encoding='utf-8').read()
lines = src.split('\n')

print('=== CHART/IMAGE usage ===')
for pat in ['add_picture', 'add_chart', 'from pptx.chart', 'XL_CHART', 'PieChart', 'BarChart', 'LineChart', 'CategoryChartData', 'add_table', 'shape.fill']:
    hits = [i+1 for i, l in enumerate(lines) if pat in l]
    print(f'{pat}: {hits[:10]}')

print('\n=== Slide size setup ===')
for i, l in enumerate(lines[:214], 1):
    if 'Cm(' in l or 'Inches(' in l or 'slide_width' in l or 'slide_height' in l or 'Presentation()' in l or 'blank' in l.lower() or 'slide_layout' in l:
        print(f'{i}: {l.strip()}')

print('\n=== _slide_body function (263-331) ===')
for i in range(262, 331):
    print(f'{i+1}: {lines[i]}')
