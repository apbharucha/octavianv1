import subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

# Check show_octavian_chatbot around query processing
print("=== How show_octavian_chatbot sends queries ===")
print(sh("grep -n -B10 -A5 'process_unbiased_query\|process_enhanced_query\|chatbot\.' ai_chatbot.py | grep -A2 -B2 'response =' | head -30"))

# Check if show_octavian_chatbot instantiates a chatbot
print("=== Chatbot instantiation ===")
print(sh("grep -n 'get_chatbot\|OctavianEnhancedChatbot' ai_chatbot.py | head -10"))

# Check if asyncio.run pattern is used to call the async methods
print("=== asyncio.run call pattern ===")
print(sh("grep -n 'asyncio.run\|get_event_loop\|run_until_complete' ai_chatbot.py | head -10"))