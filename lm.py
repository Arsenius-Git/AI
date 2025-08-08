from transformers import pipeline

classifier = pipeline("zero-shot-classification", model = "joeddav/xlm-roberta-large-xnli")
user_message = "работаете ли вы в воскресенье?"
# labels
candidate_labels = ["greeting","make an appointment","information about appointment date","other information about payment","complain","other"]

result = classifier(user_message, candidate_labels)
print(f"intent classified: {result['labels'][0]} with score {result['scores'][0]:.3f}")