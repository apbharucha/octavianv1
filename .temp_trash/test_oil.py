import asyncio
from typing import *
from ai_chatbot import OctavianEnhancedChatbot
bot = OctavianEnhancedChatbot()
symbols = bot._extract_symbols("what is happening to the oil prices right now")
print("Symbols:", symbols)
