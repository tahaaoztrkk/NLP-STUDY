# Multimodal Question Answering over Scientific Papers

Semester project for **CNG479 – Applied Natural Language Processing & Large Language Models** (Fall 2026), instructor Assist. Prof. Rezan Bakır.

The project studies question answering over scientific papers whose answers depend on figures and tables, using the [SPIQA](https://huggingface.co/datasets/google/spiqa) dataset. It asks how much a system gains from using visual information and from first retrieving the figure that a question refers to.

## Team

- Zeynep Yağmur Çalta
- Polat Elmas
- Taha Seyfülislam Öztürk

## Research question

When answering questions about figures and tables in scientific papers (SPIQA), how does answer quality change between text-only models and models that also see the image, does the gap depend on the figure type, and how much does a retrieval step that first finds the relevant figure change the result?

- **RQ1:** How accurately do text-based representations retrieve the figure a question refers to, and for which figure types do they struggle?
- **RQ2:** How does answer quality differ between text-only input and image + caption input?
- **RQ3:** How strongly does retrieval quality affect answer quality?
- **H1:** Text retrieval works well for information-dense table captions but poorly for plots and photographs with short or generic captions, where visual information is needed.

Planned evidence: Recall@k and MRR for retrieval; ROUGE-L, BERTScore and L3Score for answers.

## Dataset

**SPIQA** (Pramanick, Chellappa & Venugopalan, NeurIPS 2024 Datasets & Benchmarks; [arXiv:2407.09413](https://arxiv.org/abs/2407.09413)).

The notebooks currently use only the `test-A` metadata file (`test-A/SPIQA_testA.json`, about 0.8 MB: 118 papers, 666 questions, 1,115 figure and table captions). It is downloaded automatically with `hf_hub_download`. The full release with images is about 33 GB and is not needed yet.

Do not use `load_dataset("google/spiqa")`: the dataset viewer copy is a preview and leaves `all_figures` empty.

## Repository structure

```
NLP-STUDY/
├── notebooks/
│   └── 01_lab_preprocessing_tokenization.ipynb
├── archive/
│   └── SPIQA_adim2_adim3_onisleme_tokenizasyon.ipynb
├── requirements.txt
└── README.md
```

## Notebooks

| # | Notebook | Content |
|---|---|---|
| 01 | [`notebooks/01_lab_preprocessing_tokenization.ipynb`](notebooks/01_lab_preprocessing_tokenization.ipynb) | Text and label audit, cleaning, stemming and lemmatization, sentence segmentation, word / character / subword tokenization, BPE by hand, and figure-type classification from captions (plot, schematic, photograph, table) with TF–IDF and MiniLM embeddings on a paper-grouped split |

Notebooks are numbered in run order; later stages of the project will be added as `02_…`, `03_…`.

### Results so far (notebook 01, `test-A`, one paper-grouped split, 216 test captions)

| Experiment | Macro F1 |
|---|---:|
| TF–IDF (1,2) + logistic regression, minimal cleaning | 0.712 |
| MiniLM sentence embeddings + logistic regression | 0.705 |
| TF–IDF, "Figure N:" / "Table N:" prefix removed | 0.666 |
| TF–IDF on a random caption-level split (paper overlap, for comparison only) | 0.814 |

The test set is small, so these scores are unstable. The random split scores about ten points higher because captions of the same paper appear in both training and test data; the grouped split is the one to report.

## Getting started

```bash
git clone https://github.com/tahaaoztrkk/NLP-STUDY.git
cd NLP-STUDY
pip install -r requirements.txt
```

Then open a notebook in Jupyter, VS Code or Google Colab and run it from top to bottom. The SPIQA metadata is downloaded into `spiqa_data/`, which is excluded from version control.

## Archive

`archive/SPIQA_adim2_adim3_onisleme_tokenizasyon.ipynb` is the team's first preprocessing and tokenization notebook, kept for reference. It is superseded by notebook 01. Its known issues, and how notebook 01 handles each:

| Issue in the archived notebook | Notebook 01 |
|---|---|
| The explanation field was read as `rationale`, so explanations were never loaded (the field is called `explanation`) | Reads the field names from the file and loads `explanation` |
| The OOV comparison was not meaningful: the vocabulary cap (8,000) exceeded the vocabulary size (about 5,800), and the BPE side was reported as 0% without being measured | Builds the vocabulary on training papers and measures OOV on held-out test papers |
| The custom BPE tokenizer was trained without an initial byte alphabet, so characters unseen in training could become unknown tokens | Trains BPE with `initial_alphabet=pre_tokenizers.ByteLevel.alphabet()` |
| The sentence-segmentation example presented Punkt as handling abbreviations, but Punkt still split after "Fig.", "et al." and similar | Not fixed; documents the limitation in Checkpoint 7 |
