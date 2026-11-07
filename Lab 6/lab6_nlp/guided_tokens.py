import re

text = "I can't submit CS4944 late!"
whitespace = text.split()
words = re.findall(r"\b\w+\b", text)
print("Whitespace:", whitespace, len(whitespace))
print("Word-like:", words, len(words))
print("Original:", text)
