# LLM-SCRATCH

This repository contains code transcribed by daiyada based on the book [“Build a Large Language Model (From Scratch)”](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167?crid=228R4JI0P0QFR&dib=eyJ2IjoiMSJ9.XvZyIer9iV133BWXqNiVt_OOJXZheO54dvZtQly8MC25PNYZrN3OWsGLjbg3I0G9hI3LkjwhsORxvHIob3nvCZFgdSSQEFe07VkehijGxT03n4Amdw7lnXxnsOUuWXeglfHnewCcV3DjL9zWHELfh5DG1ZErzFym3S6ZxSuFzNvoPkaq0uDlD_CKwqHdC0KM_RdvIqF0_2RudgvzRli0V155KkusHRck3pG7ybp5VyqKDC_GgL_MEywLwLhFgX6kOCgV6Rq90eTgSHFd6ac8krpIYjsHWe6H3IXbfKGvMXc.473O1-iUZC0z2hdx8L5Z5ZTNxtNV9gNPw_mE7QZ5Y90&dib_tag=se&keywords=raschka&qid=1730250834&sprefix=raschk,aps,162&sr=8-1&linkCode=sl1&tag=rasbt03-20&linkId=84ee23afbd12067e4098443718842dac&language=en_US&ref_=as_li_ss_tl). 

Github repository related to the book is [here](https://github.com/rasbt/LLMs-from-scratch?tab=readme-ov-file).

## Initialization

## Execution

## GPTModel Parameter Mapping Table

### Overview

Different GPT implementations and papers often use different names for the same concepts.
This document summarizes which variable names are equivalent and which represent different concepts in Transformer/GPT architectures.

### 1. Tensor Shapes & Basic Dimensions

| Concept                  | Variable Name    | Common Term                  | Role                                           |
| ------------------------ | ---------------- | ---------------------------- | ---------------------------------------------- |
| Batch size               | `b`              | `batch_size`                 | Number of samples processed per forward pass   |
| Actual number of tokens  | `num_tokens`     | `seq_len`                    | Actual input sequence length (token count)     |
| Maximum number of tokens | `context_length` | `max_seq_len` / `block_size` | Maximum sequence length supported by the model |


### 2. Embedding / Model Dimensions

| Concept                    | Variable Name     | Common Term              | Role                                               |
| -------------------------- | ----------------- | ------------------------ | -------------------------------------------------- |
| Embedding size (input dim) | `emb_dim`, `d_in` | `d_model`, `hidden_size` | Dimensionality of token embeddings                 |
| Attention output dimension | `d_out`           | `d_model`                | Output dimension of Q/K/V and Multi-Head Attention |
| Per-head dimension         | `head_dim`        | `d_model / num_heads`    | Dimensionality handled by each attention head      |

### 3. Multi-Head Attention Structure

| Component                     | Variable Name                   | Common Term | Role                                                   |
| ----------------------------- | ------------------------------- | ----------- | ------------------------------------------------------ |
| Q projection input dimension  | `d_in`                          | `d_model`   | Input dimensionality to the Q projection               |
| Q/K/V linear output dimension | `d_out`                         | `d_model`   | Output of the projection before being split into heads |
| Number of heads               | `num_heads`                     | `num_heads` | Number of attention heads                              |
| Dimension per head            | `head_dim = d_out // num_heads` | `head_dim`  | Size of each head’s subspace                           |

### 4. Attention Computation Tensors

| Concept          | Variable Name  | Shape                       | Role                                           |
| ---------------- | -------------- | --------------------------- | ---------------------------------------------- |
| Attention scores | `attn_scores`  | (b, num_tokens, num_tokens) | Raw scores indicating token-to-token relevance |
| Softmax weights  | `attn_weights` | (b, num_tokens, num_tokens) | Normalized attention distribution              |
| Attention output | `context_vec`  | (b, num_tokens, d_out)      | Final contextual representation for each token |

### 5. Masking

| Concept         | Variable Name                    | Common Term   | Role                                               |
| --------------- | -------------------------------- | ------------- | -------------------------------------------------- |
| Causal mask     | `mask`                           | `causal_mask` | Prevents tokens from attending to future positions |
| Dynamic slicing | `mask[:num_tokens, :num_tokens]` | -             | Adjust mask to shorter input sequences             |

