# book-analyzer

Analyze excerpts of literature.

## Usage

1. Collect texts using [book-spider](https://github.com/EricRobertBrewer/book-spider).

2. Run `sites/db_to_csv.py`.

3. Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/). Run the server.

For example:
```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

4. Run `text/tokenize_texts.py`. Shut down the Stanford CoreNLP server.

5. Classify!
