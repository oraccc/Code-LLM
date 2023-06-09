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

<img src="https://github.com/oraccc/Code-LLM/blob/master/imgs/sentinel-tokens.png?raw=true" width="450">

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

same as code

##### Jupyter-structured

```makefile
<jupyter_start><jupyter_text>TEXT<jupyter_code>CODE
<jupyter_output>OUTPUT<jupyter_text> ...
```

##### Git commits

```makefile
<commit_before>code<commit_msg>text<commit_after>code<eos>
```



#### Tokenizer

We use the Hugging Face Tokenizers library to train a **byte-level Byte-Pair-Encoding** with a vocabulary size of **49,152** tokens—including the sentinel tokens



#### Model Architecture

We trained a **15.5B parameter** model with the same architecture as SantaCoder. It is a **decoder-only** Transformer with **Fill-in-the-Middle** (FIM; Bavarian et al., 2022), **MultiQuery-Attention** (MQA; Shazeer, 2019), and learned absolute positional embeddings. 

Use **FlashAttention** (Dao et al., 2022) to speed up the attention computation and reduce its memory footprint, allowing us to scale to context length **8K**. To make FlashAttention work with MQA during training, we simply expand the key and value before calling the attention kernel. 

<img src="https://github.com/oraccc/Code-LLM/blob/master/imgs/starcoder.png?raw=true" width="350">



#### Training Details

**StarCoderBase** 

* model was trained for **250k** iterations
* a batch size of **4M tokens**, for a total of one trillion **tokens**
* used Adam (Kingma & Ba, 2015) with $\beta_1= 0.9, \beta_2 = 0.95, \epsilon= 10^{−8}$ and a weight decay of 0.1
* the learning rate followed a cosine decay from $3 × 10^{−4}$ to $3 × 10^{−5}$ after a linear warmup of **2,000** iterations.

**StarCoder**

* fine-tuned a Python variant of the model for 2 epochs on the Python subset of the training data. 
* used the same settings as StarCoderBase
* decreased the learning rate to $5 × 10^{−5}$ and decayed it to $5 × 10^{−6}$ after **1,000** iterations of linear warmup
* trained for **8,500** steps



#### Multi-node GPU Setup

trained our model on a GPU cluster with 512 A100 80 GB GPUs distributed across 64 nodes

partitioned the model with a 3D-parallel layout that shards the model with both tensor and pipeline parallelism rank 4, requiring 16 GPUs (two nodes) for one replica



### V: Evaluation

Dataset: HumanEval, MBPP, DS-1000

The BigCode Evaluation Harness

#### StarCoder: Python Evaluation

##### HumanEval & MBPP

HumanEval and MBPP are widely-used benchmarks for Code LLMs that consist of **hundreds of Python programming problems that use test cases** to validate the code produced by a Code LLM

We report performance using the **pass@k metric**: a benchmark problem is considered solved if any one of k code samples passes every test case. 

we use sampling temperature 0.2 for pass@1, and temperature 0.8 for k > 1. 

We generate n = 200 samples for all experiments with open-access models. 

For API models, we use n = 20 samples, which is enough to estimate pass@1. 

We focus on the simplest version of pass@k, which is **pass@1**: the likelihood that a problem is solved in a single attempt by the model.

We found that the **following prefix at temperature 0.1 boosts performance** on HumanEval to 40.82%

```makefile
<filename>solutions/solution_1.py
# Here is the correct implementation of the code exercise
```

##### DS-1000 Python Data Science Benchmark

the DS-1000 benchmark (Lai et al., 2022) has a suite of 1,000 **realistic and practical data science workflows** across seven libraries and evaluates generations in execution against test cases



#### StarCoder and StarCoderBase: Multi-language Evaluation

This section also shows that StarCoder, despite being fine-tuned on Python, remains a very capable multi-language Code LLM and even outperforms StarCoderBase on some languages
