import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

print("=" * 60)
print("A. show_octavian_chatbot: which process method does it call?")
print("=" * 60)
print(sh("grep -n -A10 'process_enhanced_query\\|process_unbiased_query' ai_chatbot.py | grep -A2 'show_octavian' | head -20"))
print(sh("grep -n 'process_enhanced_query\\|process_unbiased_query' ai_chatbot.py | grep -v 'def process\\|_generate\\|^$' | head -20"))

print("=" * 60)
print("B. show_octavian_chatbot main flow around the query submission")
print("=" * 60)
print(sh("grep -n 'with st.spinner\\|response = await\\|response = asyncio\\|process_enhanced' ai_chatbot.py | head -20"))
print(sh("sed -n '5570,5615p' ai_chatbot.py"))

print("=" * 60)
print("C. How charts render after getting response")
print("=" * 60)
print(sh("sed -n '5600,5660p' ai_chatbot.py"))