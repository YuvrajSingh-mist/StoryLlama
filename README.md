
# Introducing StoryLlama - A Smaller Language Model for Bedtime Stories! 

- So, I trained a Llama a 88M architecture I coded from ground up to build a small instruct model, going through the below-mentioned stages from scratch.
- Trained on TiyStories dataset form HuggingFace consisting of 4B tokens for a total of 5000 steps



 ###  Pretraining

#### Dataset

 - I used the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset from HuggingFace.

  1) Train dataset - 2 M records approx
  2) Val dataset - 26K records approx



---

####  ModelArgs (Hyperparameters)

# Model Configuration

Below is a table summarizing the configuration parameters for the model:

| Parameter                      | Description                                                                 | Default Value                     | Type      |
|--------------------------------|-----------------------------------------------------------------------------|-----------------------------------|-----------|
| `epochs`                       | Number of training epochs                                                   | `4`                               | `int`     |
| `block_size`                   | Size of each block (context length)                                         | `512`                             | `int`     |
| `batch_size`                   | Batch size for training                                                    | `64`                              | `int`     |
| `inference`                    | Inference mode (not specified)                                              | `None`                            | `None`    |
| `embeddings_dims`              | Dimensionality of embeddings                                                | `512`                             | `int`     |
| `attn_dropout`                 | Dropout rate for attention layers                                           | `0.1`                             | `float`   |
| `no_of_heads`                  | Number of attention heads                                                   | `8`                               | `int`     |
| `dropout`                      | Dropout rate for the model                                                  | `0.1`                             | `float`   |
| `val_epochs`                   | Number of validation epochs                                                 | `2`                               | `int`     |
| `max_lr`                       | Maximum learning rate                                                       | `6e-4`                            | `float`   |
| `no_of_decoder_layers`         | Number of decoder layers                                                    | `8`                               | `int`     |
| `weight_decay_optim`           | Weight decay for the optimizer                                              | `0.1`                             | `float`   |
| `beta_1`                       | Beta 1 for Adam optimizer                                                   | `0.9`                             | `float`   |
| `beta_2`                       | Beta 2 for Adam optimizer                                                   | `0.95`                            | `float`   |
| `clip`                         | Gradient clipping value                                                     | `1.0`                             | `float`   |
| `device`                       | Device to run the model (`cuda` or `cpu`)                                   | `'cuda'`                          | `str`     |
| `no_kv_heads`                  | Number of key-value heads                                                   | `2`                               | `int`     |
| `vocab_size`                   | Size of the vocabulary                                                      | `50304`                           | `int`     |
| `eps`                          | Epsilon value for numerical stability                                       | `1e-5`                            | `float`   |
| `dtype`                        | Data type for tensors (`bfloat16` if supported, else `float16`)             | `'bfloat16'` or `'float16'`       | `str`     |
| `save_checkpoint_dir`          | Directory to save model checkpoints                                         | `"checkpoints"`                   | `str`     |
| `prompt`                       | Default prompt for inference                                                | `"Once upon a time"`              | `str`     |
| `save_checkpoint_iter`         | Save checkpoint every N iterations                                         | `50`                              | `int`     |
| `total_iters`                  | Total number of training iterations                                        | `10000`                           | `int`     |
| `eval_iters`                   | Evaluate model every N iterations                                          | `50`                              | `int`     |
| `eval_check`                   | Check evaluation metrics every N iterations                                | `100`                             | `int`     |
| `warmup_iters`                 | Number of warmup iterations for learning rate scheduling                   | `700`                             | `int`     |
| `min_lr`                       | Minimum learning rate (10% of `max_lr`)                                     | `0.1 * max_lr`                    | `float`   |
| `lr_decay_iters`               | Number of iterations for learning rate decay                               | `10000`                           | `int`     |
| `total_batch_size`             | Total batch size across all devices                                         | `524288`                          | `int`     |
| `micro_batch_size`             | Micro batch size per device                                                | `batch_size`                      | `int`     |
| `gradient_accumulation_steps`  | Gradient accumulation steps                                                 | `total_batch_size // (micro_batch_size * (block_size * torch.cuda.device_count()))` | `int` |
---
#### Hardware Setup

 - Used DPP using Pytorch torchrun consisting of 4x GeForce RTX 4090s (24gb VRAM each) rented on runpod.io
 - The model is a 1.5GB in size but needs around 5 GB of VRAM when loaded in fp32 precision
---

#### Frameworks:
**Pytorch**


--- 

#### Epochs/Steps
- Iterations (train) = 45k

- Val iterations = every 1k
---

#### Losses
- Train loss - 3.96

- Val loss - 4.01

---

#### Screenshots of the loss curves

- Epoch 1 with CosineAnnealingWarmRestarts

![Epoch 1 with CosineAnnealingWarmRestarts](images/epoch_1.jpg)

- Epoch 2 with CosineAnnealing (checkpoint from epoch 1)

![Epoch 2 with CosineAnnealing (checkpoint from epoch 1)](images/epoch_2.jpg)

- Epoch 3 with CosineAnnealing (checkpoint from epoch 2)

![Epoch 3 with CosineAnnealing (checkpoint from epoch 2)](images/epoch_3.jpg)

--- 
#### Output

- Prompt: It was a difficult time for me

![Prompt: It was a difficult time for me](images/prompt1.jpg)

- Prompt: My work life

![Prompt: My work life](images/prompt2.jpg)

---

### Local setup


### Requirements



```python
git [clone the repo](https://github.com/YuvrajSingh-mist/SmolLlama.git)
cd SmolLlama
bash ./install.sh

```
- A wandb.ai account for plotting graphs for your loss curves

- On your terminal run
```python
wandb login
```

- Enter the api key and follow the instructions and once you are succesfully logged in follow the given steps


- Download the model

```python
python donwload_model_weight.py
```


---

### Running 


#### Training a model

- Kindly hange 'device' to any of your available cuda gpus.

To run:

```python
torchrun --standalone --nproc_per_node=gpu llama.py \   
    --epochs 10 \
    --block_size 256 \
    --batch_size 32 \
    --embeddings_dims 1024 \
    --no_of_heads 8 \
    --max_lr 3e-4 \
    --prompt "Once upon a time" \
    --max_length 100 \
    --temperature 0.8
```
--standalone - if all the gpu are on one server
--npro_per_node - number of gpus available and use the keyword gpu to use all

#### Inference on a model

```python 
python inference.py --prompt "Once upon a time" --max_length 100 --temperature 0.8 --repetition_penalty 1.5 
```

