# Retrieval Augmentation for Commonsense Reasoning with Deduplication

Retrieval Augmentation for Commonsense Reasoning: A Unified Approach [arXiv](https://arxiv.org/abs/2210.12887).

1.  Download the Commonsense Corpus

    Corpus (20M): Google drive [link](https://drive.google.com/drive/folders/1oj2POBBy8kyBFNU5nHb05wu2DlcOfGnV?usp=share_link)

    -   Construct the corpus
    -   Deduplicate the corpus
    -   Modify the retrieval corpus/file path
        [conf/ctx_sources/default_sources.yaml]

    ```
    dpr_wiki:
        _target_: dpr.data.retriever_data.CsvCtxSrc
        file: path/to/corpus
        id_prefix: 'wiki:'
    ```

2.  Training the Commonsense Retriever

    Official DPR code [link](https://github.com/facebookresearch/DPR)

    RACo Training Data: Google drive [link](https://drive.google.com/drive/folders/1abY1yMj9ygF7Plb52sDEBsKqn4GlwlBx?usp=share_link)

    -   Modify the training data/file path
        [conf/datasets/encoder_train_default.yaml]

    ```
    raco_train:
        _target_: dpr.data.biencoder_data.JsonQADataset
        file: {your folder path}/train.json

    raco_dev:
        _target_: dpr.data.biencoder_data.JsonQADataset
        file: {your folder path}/dev.json
    ```

3.  Inference: Retrieve Documents

    Inference Data: Google drive [link](https://drive.google.com/drive/folders/1VMpi4hl1VYuaBPhC3gB4PDTVlXdl-6Sn?usp=share_link)

    -   Modify the inference data path
        [conf/datasets/retriever_default.yaml]

    ```
    {dataset}_train:
        _target_: dpr.data.retriever_data.CsvQASrc
        file: {your folder path}/{dataset}/train.tsv

    {dataset}_dev:
        _target_: dpr.data.retriever_data.CsvQASrc
        file: {your folder path}/{dataset}/dev.tsv

    {dataset}_test:
        _target_: dpr.data.retriever_data.CsvQASrc
        file: {your folder path}/{dataset}/test.tsv
    ```

4.  Evaluate: Retrieved Documents

## Poster

![Poster](https://raw.githubusercontent.com/swislar/RACo-Deduplication/refs/heads/main/assets/04-RAG_Info_Deduplication.jpg)


## References:

https://github.com/wyu97/RACo

https://github.com/facebookresearch/DPR
