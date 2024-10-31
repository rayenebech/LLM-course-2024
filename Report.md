# **Lab 01 : Tokenizers**
## 1. What are tokenizers?
A tokenizer is a method of breaking words into smaller parts called `tokens`. In English, suffixes, prefixes, and word stems can be examples of tokens. These tokens can be mapped into numerical values called `token ids` using a lookup table based on the language's vocabulary. When training a tokenizer from scratch, the vocabulary is built based on the tokenizer's algorithm and the training corpus. Some people refer to the words that remain as whole words after tokenization as tokens, and the parts of words that were divided as sub-tokens.

## 2. Why are they important for language modeling and LLMs?
Tokenization is a necessary step for any language model. Here are some of the most important aspects of tokenization for language models:

- **Machine's Language**: Machines cannot understand human language. The tokenizers translate a sequence of words into numerical IDs that can be fed to the language models.

- **Causal Language Modeling**: The whole mechanism of language models is based on predicting the next token given a sequence of previous tokens. Language models cannot predict a sentence as a whole at once; rather, they predict tokens sequentially based on the conditional probabilities of the previous tokens.

- **Pattern Recognition**: Some tokens can appear in more than one word, helping the models to construct a semantic pattern to understand words and build semantic relationships between words, such as words derived from the same root, verbs, adjectives, etc.

- **Self-Attention**: The success of LLMs is powered by the self-attention mechanism. LLMs compute the attention scores based on the token IDs produced by the tokenizer. For example, an LLM computes the scores of the sub-token "ing" in "language modeling." Poor tokens result in misleading attention scores.

- **Word Embeddings**: Large sequences of words can be divided into smaller tokens that models can learn meaningful embeddings from. With tokenization, each token can have its own embedding vector that represents its meaning in the semantic space. Different operations, such as semantic similarity, can be performed based on the embeddings of each token.

- **OOV Problem**: The vocabulary of language models, no matter how big it is, remains limited by a constant size. If a new word comes that does not exist in the vocabulary, the model will not be able to construct meaningful embeddings for it (Out-of-Vocabulary). However, with many tokenizers today (sub-word tokenizers), this problem is mitigated by breaking the new word into known tokens. If no known token is found in the vocabulary, the new word can be decomposed into a sequence of characters but never remains an unknown entity for the model.

## 3. What different tokenization algorithms are there and which ones are the most popular ones and why?

There are different types of tokenizers based on the algorithms used to break down a sequence of words into tokens. We can discuss three main tokenization methods:

1. **Word Tokenization**: Simply breaks down a sentence into words by splitting based on a certain separator (usually whitespace).
    - **Pros**: 
        - Easy to implement.
        - Requires no training.
        - Fast in computations.
    - **Cons**: 
        - Presents a challenge for agglutinative languages like Finnish and Turkish as one word can contain many information that need to be seperately handled by the model.
        - Results in a very large vocabulary size (if not infinite), as every word is an element of the vocabulary, and all possible derivations of a word are also new words in the vocabulary.

2. **Character Tokenization**: Breaks down a sentence into individual characters.
    - **Pros**:
        - Vocabulary size is small and equal to the number of characters in that language.
        - Useful for languages that do not have clear word boundaries, like some Asian languages (Chinese, Japanese...).
        - Easy to implement.
        - Requires no training.
        - Fast in computations.
    - **Cons**: 
        - It can be hard to capture semantic patterns between words. more effort is required for language models to be able to capture the co-occurence of some characters and the linguistic patterns.[^1]
        - The number of output tokens becomes very large when dealing with long texts. This may lead to information loss as it causes a challenge for language models as they have limited size input length.[^1]

3. **Sub-word Tokenization**: Breaks down sentences into tokens (either whole words or parts of words). It can be seen as a balance between word and character tokenization.
    - **Pros**: 
        - These tokenizers are built based on the linguistic characteristics of languages, which increases the probability of having meaningful tokens and sub-tokens that actually make sense. (If we look at the vocabulary of some tokenizers, we can see known prefixes, suffixes, and stems.)
        - Increases the ability of language models to build semantic patterns and understand the linguistic rules of a specific language.
        - Good at handling the OOV problem. 
    - **Cons**:   
        - Depends heavily on the training corpus. Having poor data, such as spelling errors, may result in a malformed vocabulary.
        - Struggles to represent rare words.
        - Difficulty in handling large numbers thus affecting the performance of large language models.[^2]



**The most commonly used tokenizers today are the sub-word tokenizers**, thanks to their ability to capture linguistic patterns that are crucial for language models. Some of the most popular ones are:

1. **Byte Pair Encoding (BPE)**: This algorithm starts by breaking down a corpus into words (this operation is known as pre-tokenization). After that, the words are decomposed into pairs of characters, and the frequency of each pair inside the corpus is computed. The most frequent pairs are added to the vocabulary as new elements. This process is repeated, and characters are merged sequentially based on their frequencies until a specific vocabulary size is reached.  
Byte-Level BPE is a popular tokenizer used by many LLMs today. 
The byte-level BPE is a subset of the BPE which uses bytes instead of characters. This is useful especially for a lot of unicode characters.

2. **WordPiece**: This method is similar to BPE in its working process. However, instead of merging pairs based on their frequencies inside the corpus, the algorithm maximizes the likelihood of the training data by computing the conditional probability of the pairs, dividing the frequency of the pair tokens appearing together by the frequencies of each token. 
    `score=(freq_of_pair)/(freq_of_first_element×freq_of_second_element)`

It is used by some auto-encoder models like BERT, DistilBERT, and ELECTRA.


3. **SentencePiece**: One problem that faces other algorithms is that they first pre-tokenize the text into a sequence of words (using white spaces generally) then tokenize the words further into tokens. However, this may cause problems as whitespace is not necessarily the real separator of words. Some languages, like Chinese, Korean and Japanese are non-segmented languages. As a solution, the SentencePiece tokenizer treats the text as a raw text stream composed of both spaces and characters. It then applies either the BPE or WordPiece algorithm.[^3] Many well-known models today, like LLaMA and Mistral, use the SentencePiece BPE tokenization method. 
A recent study[^4] showed that the BPE tokenizer implemented by the SentencePiece library outperformed the BPE implemented by the [Huggingface library](https://huggingface.co/)




[^1]: Toraman, C., Yilmaz, E. H., Şahinuc, F., & Ozcelik, O. (2023). Impact of tokenization on language models: An analysis for Turkish. ACM Transactions on Asian and Low-Resource Language Information Processing, 22(4), Article 116. https://doi.org/10.1145/3578707

[^2]: https://www.linkedin.com/pulse/exploring-byte-pair-encoding-bpe-premai-znv8f/

[^3]: https://huggingface.co/docs/transformers/en/tokenizer_summary#summary-of-the-tokenizers


[^4]: Ali, M., Fromm, M., Thellmann, K., Rutmann, R., Lübbering, M., Leveling, J., Klug, K., Ebert, J., Doll, N., Buschhoff, J. S., Jain, C., Weber, A. A., Jurkschat, L., Abdelwahab, H., John, C., Suarez, P. O., Ostendorff, M., Weinbach, S., Sifa, R., … Flores-Herr, N. (2023). Tokenizer Choice For LLM Training: Negligible or Crucial? (Version 4). arXiv. https://doi.org/10.48550/ARXIV.2310.08754
