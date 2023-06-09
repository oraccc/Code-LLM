## StarCoder Note-Taking

> :page_with_curl: [Paper Link](https://arxiv.org/abs/2305.06161)

### I: Intro

**StarCodeBase**: trained on **1 trillion** tokens, **80+** programming languages, GitHub issues, Git commits, and Jupyter notebooks from **The Stack**

**StarCode**: fine-tune StarCoder on **35B** Python tokens



The Stack ([Kocetkov et al., 2022](https://arxiv.org/abs/2211.15533)): a **6.4 TB** dataset of permissively licensed source code in **384 **programming languages, and included **54 GB **of GitHub issues and repository-level metadata in the v1.2 version of the dataset

Both StarCoder models come with a novel combination of architectural features

- an 8K context length 
- infilling capabilities through Fill-in-the-Middle 
- fast large-batch inference through Multi-Query-Attention



### II: Data Curation and Cleaning

cleaning the data by combining heuristic filtering and manual inspection

#### Programming Languages

Stack: 358 -> StarCoderBase: 86

#### GitHub Issues

remove: auto-generated, bots, poor quality, non-English

#### Weighing of Data Source

follow the **natural distribution** of data during training and sample data sources proportionally to their volume



### III: PII Redaction

#### StarEncoder

an **encoder-only** model (i.e., bi-directionally self-attentive Transformers) that can be efficiently fine-tuned for both code- and text-related tasks

leverage the **Masked Language Modelling (MLM)** and **Next Sentence Prediction (NSP)** objectives from BERT and predicted masked-out tokens from an input sentence and whether a pair of sentences occur as neighbors in a document.

training objective: $\mathcal{L} = \mathcal{L}_{MLM} + \mathcal{L}_{NSP}$

##### architecture

<img src="https://github.com/oraccc/Code-LLM/blob/master/imgs/starencoder.png?raw=true" width="350">

##### training strategy

Special tokens are added to separate code snippets and represent each input as follows: 
$$
[CLS]\{Snippet1\}[SEP]\{Snippet2\}[SEP]
$$
**NSP**: Two code snippets are selected randomly, and a decision is made on-the-fly as to whether the two pieces of code are neighbors from the same source file or are picked from two distinct documents. $L_{NSP}$ is computed using a linear classifier trained on top of the representations output at the **[CLS] special token**.

**MLM**: Tokens are masked out independently with a probability of 15%, and the results define the input-output pairs used to compute $L_{MLM}$ . 

##### training details

**100,000 steps** with a global batch size of **4,096** sequences of a maximum length of **1,024** so that approximately **400B** tokens are observed

two days using 64 NVIDIA A100 GPUs

#### PII Detection Model

We fine-tuned StarEncoder on the annotated PII dataset for the **Named Entity Recognition (NER)** task. We added a *linear layer* as a token classification head on top of the model, with 6 target classes: names, emails, keys, passwords, IP addresses, and usernames



### IV: Model Training

#### Data Formatting

##### Code

We prepended the repository name, file name, and the number of stars to the context of the code file.

```makefile
<reponame>REPONAME<filename>FILENAME<gh_stars>STARS\nCode<eos>
```

##### Issues

We used sentinel tokens to mark the opening and closing of an issue.

```makefile
<issue_start>title + USERID: comment<issue_comment>USERID: Comment
... <issue_closed (optional)> <eos>
```



##### Jupyter-scripts

##### Jupyter-structured

##### Git commits

