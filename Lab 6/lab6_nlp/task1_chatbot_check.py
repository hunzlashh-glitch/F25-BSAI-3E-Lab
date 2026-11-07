from guided_chatbot import build_bot, reply

engine = build_bot()
assert "it_help.txt" in reply(engine, "How do I reset my password?")
assert reply(engine, "") == "Please enter a question."
assert "No matching FAQ" in reply(engine, "volcano geology")
assert "Possible matches" in reply(engine, "password recovery")
assert "library.txt" in reply(engine, "When does the library open?")
assert "it_help.txt" in reply(engine, "How do I connect to the campus Wi-Fi?")
print("FAQ behaviour checks passed")
