from lab6_search import Retriever, show_hits

records = [
    {"id": "assessment:late", "source": "handbook.txt",
     "text": "Late submission is normally penalized. Students with illness "
             "may request an extension with medical evidence before the deadline."},
    {"id": "assessment:format", "source": "handbook.txt",
     "text": "Submit the assignment as a PDF through the learning portal."}]
engine = Retriever(records)
show_hits(engine.search("illness medical evidence late submission", k=1))
