import math
from transformers import pipeline

satisfied = [
    "service was quick and helpful",
    "happy with customer support",
    "support team was polite",
    "good service experience",
    "problem solved quickly"
]

unsatisfied = [
    "very slow response",
    "issue not resolved",
    "poor customer care",
    "worst support ever",
    "not helpful at all"
]

def word_count(data):
    d = {}
    for sentence in data:
        for word in sentence.split():
            d[word] = d.get(word, 0) + 1
    return d

sat_words = word_count(satisfied)
unsat_words = word_count(unsatisfied)

total_docs = len(satisfied) + len(unsatisfied)

p_sat = len(satisfied) / total_docs
p_unsat = len(unsatisfied) / total_docs

vocab = set(sat_words) | set(unsat_words)
V = len(vocab)

sat_total = sum(sat_words.values())
unsat_total = sum(unsat_words.values())

def predict_nb(text):
    s_score = math.log(p_sat)
    u_score = math.log(p_unsat)

    for word in text.split():
        s_score += math.log((sat_words.get(word, 0) + 1) / (sat_total + V))
        u_score += math.log((unsat_words.get(word, 0) + 1) / (unsat_total + V))

    return "Satisfied" if s_score > u_score else "Unsatisfied"

tests = [
    "quick and helpful service",
    "poor customer support",
    "support team was polite",
    "very slow response",
    "not helpful at all",
    "happy with the service"
]

print("Naive Bayes:\n")
for t in tests:
    print(f"{t} -> {predict_nb(t)}")

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("\nHugging Face:\n")
for t in tests:
    result = classifier(t)[0]
    label = "Satisfied" if result["label"] == "POSITIVE" else "Unsatisfied"
    print(f"{t} -> {label} ({result['score']:.2f})")

print("\nComparison:")
print("Feedback".ljust(30), "NB".ljust(12), "HF")

for t in tests:
    nb = predict_nb(t)
    hf = "Satisfied" if classifier(t)[0]["label"] == "POSITIVE" else "Unsatisfied"
    print(t.ljust(30), nb.ljust(12), hf)
