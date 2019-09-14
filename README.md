# book-analyzer

Analyze excerpts of literature.

## Usage

1. Collect texts using [book-spider](https://github.com/EricRobertBrewer/book-spider).

2. Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/). Run the server.

For example:
```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

3. Run `text/tokenize_texts.py`. After it finishes, you can shut down the Stanford CoreNLP server.

4. Run `sites/db_to_csv.py`.

5. Run `sites/bookcave/bookcave_ids.py`. This will generate a list of book IDs from the BookCave database whose text will be used for classification.

6. Run `text/generate_data.py`. This will generate pre-tokenized tensors for sentences of books, pre-tokenized embedding matrices, and categorized label tensors, all of which can be reused amongst different classifiers.

7. Classify!
