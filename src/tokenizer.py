import json
import requests

properties = {
    "outputFormat":"json",
    "annotators":"tokenize,ssplit",
    "encoding":"utf-8",
    "tokenize.options": "untokenizable=noneKeep",
    "ssplit.eolonly": "true"
}

non_batch_props = {
    "outputFormat":"json",
    "annotators":"tokenize, ssplit, lemma",
    "encoding":"utf-8",
    "tokenize.options": "untokenizable=noneKeep"
}

url = "http://localhost:9000"

def api_call(session, data, props = properties):

        response = session.post(
            url,
            params = {
                'properties': json.dumps(props),
            },
            data = data,
            timeout=60,
        )

        response.raise_for_status()

        return response.json(strict=False)

def tokenize_all(sentences, verbose_at=100):
    """Tokenize the list of sentences one element at a time"""
    session = requests.Session()
    res = []
    
    for i in range(len(sentences)):
        sentence = sentences[i]
        result = api_call(session, sentence.encode("utf-8"), non_batch_props)

        sent = []
        for res_sent in result["sentences"]:
            for token in res_sent['tokens']:
                sent.append(token['lemma'] or token['originalText'] or token['word'])
        res.append(sent)

        if (verbose_at > 0 and (i+1) % verbose_at == 0):
            print("Finished {} row.".format(i+1))

    return res


def tokenize_all_v2(sentences, batch_size=100):
    """Tokenize the list of sentences by concatenating a batch of sentence
    with newlines and feeding it to the api.
    However, we escape characters before concatenating since there can be newlines inside the sentences.
    Therefore, this can give slightly different results"""

    session = requests.Session()
    res = []
    end_idx = 0
    end_at = len(sentences)
    i = 1
    while end_idx < end_at:
        begin_idx = end_idx
        end_idx = begin_idx + batch_size

        escaped = [sent.encode("unicode_escape") for sent in sentences[begin_idx:end_idx]]
        concat_sentences =  b"\n".join(escaped)
        result = api_call(session, concat_sentences)

        if len(result["sentences"]) != len(sentences[begin_idx:end_idx]):
            print("something went wrong")

        for res_sent in result['sentences']:
            sent = []
            for token in res_sent['tokens']:
                tok = token['originalText'] or token['word']
                sent.append(tok)
            res.append(sent)

        print("Batch {} done.".format(i))
        i += 1

    return res
