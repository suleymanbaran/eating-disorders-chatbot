# Eating-Disorder Chatbot Evaluation

A research project completed as a high-school researcher in the **Polygence** mentorship program
(2023). The question: on eating-disorder guidance questions, does a small retrieval-grounded
domain chatbot match or beat a general-purpose LLM, when both are scored against reference answers?

The finding was that grounding a 7B open-weights model in clinical guideline text beat asking
`gpt-3.5-turbo` the same questions directly — **+0.061 average F1 (0.787 vs 0.726), about +8.5%
relative** — because retrieval keeps answers anchored to the source material instead of the model's
priors.

> **Not medical advice.** This is an educational research artifact evaluating model outputs against
> a small reference set. It is not a clinical tool and was never deployed or used with people.

## Approach

Two systems answered the same 15 questions, then both were scored against the same reference answers.

### A. Retrieval-grounded chatbot (`Eating_Disorders_Chatbot_Evaluation.ipynb`)

| Stage | Choice |
|---|---|
| Source | `guidlines.pdf` — clinical guidance text, loaded with LangChain `PyPDFLoader` and extracted via `textract` |
| Chunking | `RecursiveCharacterTextSplitter`, 512 tokens with 24 overlap, measured by the Llama-2 tokenizer |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Generation | `meta-llama/Llama-2-7b-chat-hf`, 8-bit quantized (`bitsandbytes`), `bfloat16`, `max_new_tokens=512`, `do_sample=True`, `top_k=30`, temperature 0.1 |
| Chain | LangChain `RetrievalQA`, `chain_type="stuff"`, retriever `k=2`, source documents returned |
| Prompt | Llama-2 `[INST]` / `<<SYS>>` format, instructing a paragraph-length answer and to say it does not know rather than fabricate one |

### B. General-purpose baseline (`ChatGPT_Evaluation.ipynb`)

`gpt-3.5-turbo` via the OpenAI chat completions API, each question asked directly with **no
retrieval** — the "just ask the LLM" baseline.

### Evaluation

Both systems were scored on the 15 `{question, answer}` pairs in `questions.json` using
**BERTScore** (`distilbert-base-uncased`) against the reference answer, reporting precision, recall
and F1 per item, then averaged across the set.

## Results

Average over all 15 questions, from the committed notebook outputs:

| System | Precision | Recall | F1 |
|---|---|---|---|
| **RAG chatbot** (Llama-2-7b-chat + FAISS) | **0.8060** | **0.7697** | **0.7871** |
| `gpt-3.5-turbo`, no retrieval | 0.7486 | 0.7045 | 0.7257 |
| Difference | +0.0573 | +0.0652 | **+0.0614** |

The retrieval-grounded 7B model beat a much larger general-purpose model on every metric, which was
the central result of the project: for a narrow domain with authoritative source text available,
grounding beats scale.

## Files

| File | Role |
|---|---|
| `Eating_Disorders_Chatbot_Evaluation.ipynb` | Builds the RAG pipeline, answers all 15 questions, scores with BERTScore |
| `ChatGPT_Evaluation.ipynb` | Runs the `gpt-3.5-turbo` baseline and scores it the same way |
| `questions.json` | The 15 question / reference-answer pairs used for scoring |
| `guidlines.pdf` | The clinical guidance text the retriever indexes (filename as originally committed) |

## Reproducing

Both notebooks were run in Google Colab on a GPU runtime; the absolute paths (`/content/...`) reflect
that. You also need access to the gated `meta-llama/Llama-2-7b-chat-hf` weights via
`huggingface_hub.notebook_login()`.

1. Upload `guidlines.pdf` and `questions.json` to the runtime.
2. Run `Eating_Disorders_Chatbot_Evaluation.ipynb` top to bottom — it writes `ed_chatbot.csv`.
3. Run `ChatGPT_Evaluation.ipynb` with your own OpenAI key — it writes `ed_chatbot_gptAnswers.csv`.

## Limitations

Stated plainly, because they bound how far the result generalizes:

- **15 questions.** Enough to show a consistent direction, not enough for statistical significance.
- **BERTScore measures semantic similarity to a reference answer, not clinical correctness.** A
  fluent answer that agrees with the reference scores well; a clinically safer answer phrased
  differently may not. A blinded clinician rating would be the stronger instrument.
- **Single guideline source, single retriever configuration.** `k=2`, one chunk size and one embedding
  model were used without a sweep.
- **2023-era stack.** Llama-2-7b, `gpt-3.5-turbo` and the LangChain API of the time; the code is
  preserved as-run rather than updated.
