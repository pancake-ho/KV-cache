import re


def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def process_docs(dataset):
    def process_doc(document):
        context = document["ctx_a"] + " " + document["ctx_b"].capitalize()
        return {
            "query": preprocess(document["activity_label"] + ": " + context),
            "choices": [preprocess(ending) for ending in document["endings"]],
            "gold": int(document["label"]),
        }

    return dataset.map(process_doc)
