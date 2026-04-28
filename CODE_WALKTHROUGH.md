# Deep-Dive Code Walkthrough

**Reading order:** `dataloader.py` → `RedditTransformer.py` → `BertRegression.py` → `RoBERTaMTL.py` → `RoBERTaRegression.py` → `train.py`

---

## 1. `dataloader.py` — Start Here: Understanding the Data

Before any model can run, you need to understand what the data looks like and how it flows into the network. This is the right starting point.

### What does a row in the CSV actually look like?

```
body                        → The raw Reddit comment text
usVSthem_scale              → 0.2865... (float, the bias score we want to predict)
is_Disc_Crit                → True/False (is the comment discriminatory or critical?)
group                       → "Refugees" (which social group is the comment about?)
bias                        → "left" / "right" / "center" (political bias of the source)
Anger, Contempt, Disgust... → True/False (13 emotion columns)
allsides_name               → "CNN - Editorial" (the media outlet)
```

The **primary target** is `usVSthem_scale`. Everything else is optional side information for multi-task learning.

---

### `class RedditDataset(Dataset)`

```python
class RedditDataset(Dataset):
    def __init__(self, data_csv='file.csv', aux_task='group', le=None, le_aux=None):
        self.comments = pd.read_csv(data_csv)
```

A `Dataset` is a PyTorch contract. You must implement `__len__` and `__getitem__`. PyTorch's `DataLoader` then wraps it and handles shuffling, batching, and multi-process loading.

**The aux task branching logic:**

```python
aux_task_str = str(aux_task)

if aux_task_str == 'None':
    # No auxiliary task at all.
    # We still need a label_aux column to exist so the batch always has the same shape.
    # We just fill it with a dummy zero.
    self.comments['label_aux'] = 0

elif aux_task_str == 'emotions':
    # Concatenate all 13 emotion columns into a single list per row.
    # e.g.: [True, False, False, True, ...] (13 values)
    # This is a multi-label classification target.
    self.comments['label_aux'] = self.comments[[
        'Anger','Contempt','Disgust','Fear','Gratitude','Guilt',
        'Happiness','Hope','Pride','Relief','Sadness','Sympathy','Emotions_Neutral'
    ]].values.tolist()
    
    # Also save the column names so we can label confusion matrix axes later.
    self.columns = list(self.comments[[...]].columns)

else:
    # e.g. aux_task = 'group' or 'bias'
    # le_aux is a pre-fitted sklearn LabelEncoder (fitted earlier in __build_model).
    # We only TRANSFORM here (never re-fit!) because fitting must happen on the full
    # concatenated dataset (train+val+test) to guarantee consistent integer IDs.
    # If you re-fit on the train set alone, 'Refugees' might be class 3 in train
    # but class 5 in val, which would corrupt training.
    self.comments['label_aux'] = le_aux.transform(self.comments[aux_task_str].values)
```

**`__getitem__` — what one sample returns:**

```python
def __getitem__(self, idx):
    return self.comments.iloc[idx][['body', 'label_aux', 'group', 'bias', 'usVSthem_scale']]
```

This returns a pandas Series (a dict-like object). The DataLoader will call this function repeatedly for every index in the dataset. Each returned row will be gathered into a batch by the collator.

---

### `class MyCollator`

The collator is the critical bridge between raw text and tensor inputs. The DataLoader calls it with a **list** of samples (from `__getitem__`), and it must return batched tensors.

```python
class MyCollator(object):
    def __init__(self, model_name, max_length):
        # Load the tokenizer that matches the model we're using.
        # 'bert-base-uncased' → BertTokenizer
        # 'roberta-base'      → RobertaTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
```

**`__call__` — called by DataLoader with a batch of samples:**

```python
def __call__(self, batch):
    output = {}
    
    # Step 1: URL replacement
    # Reddit comments often contain URLs like https://example.com/page
    # These URLs fragment into many tokens and carry no semantic meaning.
    # We replace them all with a single special token 'LINK' to save token budget.
    texts = [
        re.sub(r'\w+:\/{2}[\d\w-]+(\.\d\w-]+)*(?:(?:\/[^\s/]*))*', 'LINK', 
               comment['body'], flags=re.MULTILINE)
        for comment in batch
    ]
    
    # Step 2: Tokenize
    # padding='longest'  → pad all sequences in this batch to the length of the longest one
    # truncation=True    → if a comment > max_length tokens, cut it off
    # max_length=512     → BERT's hard limit (it has 512 position embeddings)
    # return_tensors='pt'→ return PyTorch tensors, not lists
    # add_special_tokens → prepend [CLS] and append [SEP] automatically
    tokenized = self.tokenizer(
        texts,
        padding='longest',
        truncation=True,
        max_length=self.max_length,
        return_tensors='pt',
        add_special_tokens=True
    )
    # tokenized.data is a dict:
    #   'input_ids':      shape [batch_size, seq_len]  — integer token IDs
    #   'attention_mask': shape [batch_size, seq_len]  — 1 for real tokens, 0 for padding
    #   'token_type_ids': shape [batch_size, seq_len]  — all zeros for BERT (single sentence)
    
    # Step 3: Package labels
    # usVSthem_scale is the MAIN regression target
    output['labels'] = torch.tensor(
        [element['usVSthem_scale'] for element in batch], dtype=torch.float
    )
    # label_aux is the AUXILIARY task target (could be 0, an int, or a list)
    output['labels_aux'] = torch.tensor(
        [element['label_aux'] for element in batch], dtype=torch.float
    )
    
    # Return: (inputs_dict, targets_dict)
    # inputs_dict  → goes into model.forward()
    # targets_dict → used for loss computation
    return tokenized.data, output
```

**What does the tokenizer actually do?**

Take this comment: `"Refugees are invading us."`

After tokenization with `bert-base-uncased`:
```
[CLS]  refugees  are  invading  us  .  [SEP]  [PAD]  [PAD] ...
  101    7884    2024    9901   2149  1012  102     0      0    ...
```
- `101` = `[CLS]` — always the first token, its embedding is what the model uses as the whole-sentence representation (for BERT)
- `102` = `[SEP]` — sentence separator
- `0`   = `[PAD]` — padding; the attention mask will be 0 here so the model ignores these

---

### `sentiment_analysis_dataset()`

```python
def sentiment_analysis_dataset(hparams, train=True, val=True, test=True):
    if train:
        dataset = RedditDataset(hparams.train_csv, hparams.aux_task, hparams.le, hparams.le_aux)
    if val:
        dataset = RedditDataset(hparams.dev_csv, ...)
    if test:
        dataset = RedditDataset(hparams.test_csv, ...)
    return dataset
```

> **Note:** This function has a subtle bug — each `if` overwrites `dataset`, so if you call it with `train=True, val=True`, you only get the val set back. The callers in `BertRegression.py` always call it with exactly one `True` (e.g. `val=False, test=False`), so it works correctly in practice.

---

## 2. `RedditTransformer.py` — The Custom BERT Architecture

This is the neural network itself (the "model" in `model.forward()`). The PyTorch Lightning module in `BertRegression.py` delegates all computations here.

### Why customize BERT at all?

Standard BERT outputs **one** embedding from the `[CLS]` token, which you then pass to a single classification head. But this project does **multi-task learning**: it wants to simultaneously predict the bias score (main task) and, optionally, either the social group or the emotions (auxiliary task).

The trick used here: **fork the last transformer layer**. The idea is:
- Layers 1–11: Learn general language understanding (shared)
- Layer 12 (Main copy): Specialise for bias score prediction
- Layer 12 (Aux copy): Specialise for the secondary task

This way, the two tasks don't interfere at the final layer level, while still sharing all the lower-level features.

---

### `class RedditTransformer`

```python
class RedditTransformer(torch.nn.Module):
    def __init__(self, model_name, num_classes, extra_dropout, num_groups):
```

Parameters:
- `model_name`: `"bert-base-uncased"` — the HuggingFace pretrained model to load
- `num_classes`: always `1` — we're doing regression (predicting a single float)
- `extra_dropout`: additional dropout probability on top of BERT's default (regularization)
- `num_groups`: number of classes in the auxiliary task (e.g. 5 social groups), or `None` if no aux task

**Loading and modifying the pretrained BERT encoder:**

```python
config = AutoConfig.from_pretrained(
    model_name,
    output_hidden_states=True,
    use_cache=False,
    attn_implementation="eager"   # disables FlashAttention, avoids version conflicts
)
self.encoder = AutoModel.from_pretrained(model_name, config=config)

# Add extra dropout to every layer's attention and output
for layer in self.encoder.encoder.layer:         # 12 transformer layers
    layer.attention.self.dropout = torch.nn.Dropout(
        self.encoder.config.attention_probs_dropout_prob + extra_dropout
    )
    layer.output.dropout = torch.nn.Dropout(
        self.encoder.config.hidden_dropout_prob + extra_dropout
    )
```

**Building the classification head:**

```python
self.classification_head = torch.nn.Sequential(
    torch.nn.Dropout(config.hidden_dropout_prob + extra_dropout),
    torch.nn.Linear(config.hidden_size, num_classes),  # 768 → 1 (for BERT base)
)
```

**If an auxiliary task is active, swap out the encoder's last layer:**

```python
if num_groups is not None:
    # Replace standard BertEncoder + BertPooler with our custom versions
    self.encoder.encoder = BertEncoder(config, self.encoder.encoder.layer)
    self.encoder.pooler  = BertPooler(config, self.encoder.pooler.dense)
    self.aux = True
    
    # Auxiliary head: hidden_size → num_groups
    self.classification_head_aux = torch.nn.Sequential(
        torch.nn.Dropout(...),
        torch.nn.Linear(config.hidden_size, num_groups),
    )
else:
    self.aux = False
```

---

### `class BertEncoder`

This replaces the standard 12-layer HuggingFace BertEncoder.

```python
class BertEncoder(torch.nn.Module):
    def __init__(self, config, layers):
        super().__init__()
        # Layers 0-10 (first 11): shared, wrapped in ModuleList so PyTorch tracks them
        self.layer = torch.nn.ModuleList(layers[:-1])
        
        # Layer 11 (the 12th): deep-copied into two independent branches
        # deep copy = creates brand new weight tensors, no gradient sharing
        self.layer_main = copy.deepcopy(layers[-1])  # for bias regression
        self.layer_aux  = copy.deepcopy(layers[-1])  # for auxiliary task
```

**`forward()`:**

```python
def forward(self, hidden_states, attention_mask=None, ...):
    # 1. Run through layers 0–10 (shared)
    for i, layer_module in enumerate(self.layer):
        hidden_states, _, _ = self._layer_loop(hidden_states, layer_module, ...)
    
    # 2. Split: run the same hidden_states through BOTH layer 12 copies independently
    hidden_states_main, _, _ = self._layer_loop(hidden_states, self.layer_main, ...)
    hidden_states_aux, _, _  = self._layer_loop(hidden_states, self.layer_aux,  ...)
    
    # 3. Return as a tuple in position [0] — BertPooler expects this layout
    return ((hidden_states_main, hidden_states_aux),)
    #         ↑ shape: [B, S, H]   ↑ shape: [B, S, H]
```

At this point you have two streams of `[B, S, 768]` tensors. `S` is the sequence length, `B` is batch size, `768` is BERT's hidden dimension. One stream has seen gradients from the bias task, the other from the aux task.

---

### `class BertPooler`

Standard BERT pooling: take the `[CLS]` token hidden state (index 0 along the sequence dimension) and pass it through a linear layer and Tanh. This squashes the 768-dim vector into something representing the whole sentence.

Here we do this **twice** — once per task stream:

```python
class BertPooler(torch.nn.Module):
    def __init__(self, config, dense):
        super().__init__()
        self.dense_main = copy.deepcopy(dense)  # Linear(768, 768)
        self.dense_aux  = copy.deepcopy(dense)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        # hidden_states = (hidden_states_main, hidden_states_aux) from BertEncoder
        # Each is shape [B, S, 768]; we take position 0 = the [CLS] token
        first_token_tensor_main = hidden_states[0][:, 0]   # [B, 768]
        first_token_tensor_aux  = hidden_states[1][:, 0]   # [B, 768]
        
        pooled_output_main = self.activation(self.dense_main(first_token_tensor_main))
        pooled_output_aux  = self.activation(self.dense_aux(first_token_tensor_aux))
        
        return (pooled_output_main, pooled_output_aux)
        # Each: [B, 768] — one sentence-level vector per task stream
```

---

### `RedditTransformer.forward()`

This is the full forward pass:

```python
def forward(self, batch):
    input_ids      = batch['input_ids']       # [B, S]
    attention_mask = batch['attention_mask']  # [B, S]
    
    bert = self.encoder  # the BertModel
    
    # Step 1: Embed tokens → [B, S, 768]
    # Converts integer token IDs into continuous vectors (token + position + segment embeddings)
    embedding_output = bert.embeddings(
        input_ids=input_ids,
        token_type_ids=torch.zeros_like(input_ids),  # all zeros = single sentence
        ...
    )
    
    # Step 2: Create extended attention mask
    # Converts [B, S] binary mask → [B, 1, 1, S] additive mask
    # 0 (real token) → 0.0 in the mask (no penalty)
    # 0 (padding)    → -10000.0 (effectively -inf after softmax = ignored in attention)
    extended_attention_mask = bert.get_extended_attention_mask(
        attention_mask, input_ids.shape
    )
    
    # Step 3: Run through our custom BertEncoder (which forks at layer 12)
    encoder_outputs = bert.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        ...
        return_dict=False,
    )
    # encoder_outputs[0] = tuple of (hidden_states_main, hidden_states_aux)
    
    # Step 4: Pool (BertPooler)
    pooled_output = bert.pooler(encoder_outputs[0])
    # pooled_output = (features_main [B, 768], features_aux [B, 768])
    
    # Step 5: Classification heads
    if self.aux:
        features_main = pooled_output[0]
        features_aux  = pooled_output[1]
        logits_main = self.classification_head(features_main)      # [B, 1]
        logits_aux  = self.classification_head_aux(features_aux)   # [B, num_groups]
        return logits_main, logits_aux, encoder_outputs[0]
    else:
        features    = pooled_output
        logits_main = self.classification_head(features)           # [B, 1]
        return logits_main, None, encoder_outputs[0]
```

---

## 3. `BertRegression.py` — The BERT Training Harness (Lightning Module)

This file wraps `RedditTransformer` inside PyTorch Lightning's `LightningModule`. Lightning automates the training loop, GPU placement, logging, and checkpointing. You only define *what* happens per step; Lightning handles *when* and *how often*.

### `__init__`

```python
class BERTRegressor(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        # hparams arrives from HyperOptArgumentParser — it's a Namespace object.
        # TensorBoard can't serialize test_tube's internal objects, so we filter
        # hparams down to only primitive types (int, float, str, bool, None).
        clean_hparams = {
            k: v for k, v in vars(hparams).items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
        self.save_hyperparameters(clean_hparams)
        # After this, self.hparams is a Lightning-managed dict-like namespace.
        # It gets saved into every .ckpt file automatically.
        
        # Output accumulation lists (PL 2.0 pattern — no longer returned from steps)
        self.training_step_outputs   = []
        self.validation_step_outputs = []
        self.test_step_outputs       = []
        
        # MyCollator handles tokenization; it's initialized once and reused
        self.prepare_sample = MyCollator(self.hparams.encoder_model, self.hparams.max_length)
        
        self.__build_model()   # creates self.model = RedditTransformer(...)
        self.__build_loss()    # creates self._loss, self._loss_aux
        
        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
```

---

### `__build_model()`

```python
def __build_model(self):
    # Read all three splits at once and concatenate them.
    # WHY: We need to fit the LabelEncoder on the full label universe
    # before any training starts. If 'Muslims' only appears in the test set
    # and not in train, fitting only on train would crash at test time.
    train_df = pd.read_csv(self.hparams.train_csv)
    test_df  = pd.read_csv(self.hparams.test_csv)
    dev_df   = pd.read_csv(self.hparams.dev_csv)
    comments = pd.concat([train_df, test_df, dev_df])
    
    self.hparams.le     = LabelEncoder()   # (unused for regression target, kept for compat)
    self.hparams.le_aux = LabelEncoder()   # for the auxiliary task column
    
    aux_task_str = str(self.hparams.aux_task)
    
    if aux_task_str not in ('None', 'emotions'):
        # e.g. aux_task = 'group' → fit on all unique group strings
        # ['Conservatives', 'Immigrants', 'Jews', 'Liberals', 'Muslims', 'Refugees']
        # → encoded as [0, 1, 2, 3, 4, 5]
        self.hparams.le_aux.fit(comments[self.hparams.aux_task].values)
        
        self.model = RedditTransformer(
            self.hparams.encoder_model,
            num_classes=1,
            extra_dropout=self.hparams.extra_dropout,
            num_groups=len(self.hparams.le_aux.classes_)   # 6 in the example above
        )
        # Loss weights — initialized with a slight bias based on loss_aux_dropout param
        self.weights = nn.Parameter(
            torch.Tensor([1 + self.hparams.loss_aux_dropout,
                          1 - self.hparams.loss_aux_dropout]),
            requires_grad=True
        )
        # weights[0] controls how much the main task loss contributes
        # weights[1] controls how much the aux task loss contributes
        # Both are LEARNABLE — that's the GradNorm trick
        self.alpha = 0.5
        
    elif aux_task_str == 'emotions':
        self.model = RedditTransformer(..., num_groups=13)  # 13 emotion classes
        ...
        
    else:  # no aux task
        self.model = RedditTransformer(..., num_groups=None)
```

---

### `__build_loss()`

```python
def __build_loss(self):
    # Main task: predict a float (usVSthem_scale ∈ [0, 1])
    # MSELoss = Mean Squared Error = average of (predicted - actual)^2
    self._loss = nn.MSELoss()
    
    if self.hparams.aux_task == 'emotions':
        # Multi-label: each of the 13 emotions is independently True/False
        # BCEWithLogitsLoss = Binary Cross Entropy (applied independently per emotion)
        # "WithLogits" means no sigmoid needed before calling this — it's built in (numerically stable)
        self._loss_aux = nn.BCEWithLogitsLoss()
    else:
        # Single-label classification: predict which social group or bias category
        # CrossEntropyLoss = softmax + negative log likelihood (standard multi-class loss)
        self._loss_aux = nn.CrossEntropyLoss()
    
    # For GradNorm: measures the absolute difference between two losses (L1 = mean absolute error)
    self._gradLoss = nn.L1Loss()
```

---

### `training_step()` — Called Once Per Batch

```python
def training_step(self, batch, batch_nb):
    inputs, targets = batch
    # inputs  = {'input_ids': [B,S], 'attention_mask': [B,S], ...}
    # targets = {'labels': [B], 'labels_aux': [B] or [B, 13]}
    
    model_out = self.forward(inputs)
    # model_out = {'logits': [B,1], 'logits_aux': [B, num_groups] or None, 'hidden_states': ...}
    
    loss_val = self.loss(model_out, targets)
    # loss_val = (main_loss_tensor, aux_loss_tensor_or_None)
    
    if str(self.hparams.aux_task) != "None":
        # Stack both losses into a single tensor: [main_loss, aux_loss]
        task_losses = torch.stack(loss_val)
        # Weighted sum using our learnable weights
        total_weighted_loss = (self.weights * task_losses).sum()
    else:
        task_losses = loss_val[0]
        total_weighted_loss = task_losses
    
    self.log("train_loss", total_weighted_loss, prog_bar=True, sync_dist=True)
    self.training_step_outputs.append({"loss": task_losses})
    return {"loss": task_losses}
```

---

### `backward()` — The GradNorm Magic

This is the most complex part of the project. GradNorm is a technique that automatically balances the learning rates of different tasks so one doesn't dominate.

```python
def backward(self, loss, *args, **kwargs):
    aux_active = str(self.hparams.aux_task) != "None"
    
    if aux_active and self.hparams.gradnorm:
        # loss = [main_loss, aux_loss] (a 2-element tensor)
        
        # Weighted losses
        loss_val = self.weights * loss
        total_weighted_loss = loss_val.sum()
        
        # Step 1: Normal backprop to update all model weights
        total_weighted_loss.backward(retain_graph=True)
        # 'retain_graph=True' keeps the computation graph alive so we can
        # compute additional gradients below
        
        # Step 2: Compute GradNorm — figure out how "hard" each task is
        self.weights.grad.data.zero_()  # clear the weight gradients before recomputing
        
        # Get the parameters of the *last shared layer* (layer 11 in 0-indexed)
        W = list(self.model.encoder.encoder.layer[-1].output.parameters())
        
        norms = []
        for w_i, L_i in zip(self.weights, loss.flatten()):
            # Compute gradient of task i's loss with respect to W (last shared layer)
            gLgW = torch.autograd.grad(L_i, W, retain_graph=True)
            
            # Compute the L2 norm of that gradient (sum of squares of all gradient tensors)
            norm = sum(torch.norm(g)**2 for g in gLgW)
            norm = torch.sqrt(norm)
            norms.append(norm * w_i)   # scale by task weight
        
        norms = torch.stack(norms)
        
        # Step 3: Compute the target gradient norm (what we want each task's norm to be)
        # On the first step, record the initial losses as our baseline
        if self.trainer.global_step == 0:
            self.initial_loses = loss.detach()
        
        with torch.no_grad():
            # How much has each task's loss changed from the start?
            loss_ratios = loss / self.initial_loses
            
            # Normalize: tasks that improved more than average get a higher weight target
            inverse_train_rates = loss_ratios / loss_ratios.mean()
            
            # Target norm = mean_norm * (loss_ratio ^ alpha)
            # alpha=0.5 controls how aggressively we rebalance (0=no rebalance, 1=full)
            constant_term = norms.mean() * (inverse_train_rates ** self.alpha)
        
        # Step 4: GradNorm loss = how far are actual norms from target norms?
        grad_norm_loss = (norms - constant_term).abs().sum()
        
        # Step 5: Update task weights via this auxiliary loss
        self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]
        # This updates ONLY the weights tensor, not the model parameters
        
    elif aux_active:
        # Simple weighted loss, no GradNorm
        loss_val = self.weights * loss
        loss_val.sum().backward()
    else:
        loss.backward()
```

---

### `on_validation_epoch_end()` — Aggregating Metrics

```python
def on_validation_epoch_end(self):
    outputs = self.validation_step_outputs
    
    # Average the loss across all batches in the epoch
    val_loss_mean = torch.stack([o["val_loss"] for o in outputs]).mean()
    
    # Average the auxiliary task accuracy
    val_acc_aux_mean = torch.stack([o["val_acc_aux"] for o in outputs]).mean()
    
    # Concatenate all predictions and labels across all batches → one big (N,) tensor
    val_y     = torch.cat([o["labels"]      for o in outputs])
    val_y_hat = torch.cat([o["predictions"] for o in outputs])
    
    # Sum confusion matrices across all batches (they have the same shape)
    conf_matrix_aux = torch.stack([o["conf_matrix_aux"] for o in outputs]).sum(dim=0)
    
    # Pearson correlation: measures how well predicted values track actual values
    # r=1.0 means perfect positive correlation (model's ranking is perfect)
    # r=0.0 means no correlation (model is just guessing)
    # r=-1.0 means perfect inverse correlation
    def _pearson(x, y):
        x = x.float().flatten()
        y = y.float().flatten()
        x_mean = x - x.mean()
        y_mean = y - y.mean()
        return (x_mean * y_mean).sum() / (x_mean.norm() * y_mean.norm() + 1e-8)
        # 1e-8 prevents division by zero if predictions are constant
    
    pearsonr = _pearson(val_y, val_y_hat)
    
    # Log everything → goes to TensorBoard and the terminal progress bar
    self.log("val_loss",    val_loss_mean,    prog_bar=True, sync_dist=True)
    self.log("val_pearson", pearsonr,         prog_bar=True, sync_dist=True)
    self.log("val_acc_aux", val_acc_aux_mean, prog_bar=True, sync_dist=True)
    
    # Only rank 0 creates the confusion matrix figure (avoids duplicates in multi-GPU)
    if self.trainer.is_global_zero and str(self.hparams.aux_task) != "None":
        fig, ax = plt.subplots(figsize=(10, 7))
        sn.heatmap(conf_matrix_aux.float().cpu(), annot=True, ax=ax)
        self.logger.experiment.add_figure("confusion matrix Aux", fig)
        plt.close(fig)
    
    self.validation_step_outputs.clear()   # free memory
```

---

### `configure_optimizers()` — Learning Rate Setup

```python
def configure_optimizers(self):
    # Different learning rates for different parts of the model
    if aux_task_str != "None" and self.hparams.gradnorm:
        param_groups = [
            {"params": self.model.parameters(), "lr": self.hparams.encoder_learning_rate},
            {"params": self.weights, "lr": 1e-2},  # task weights learn faster
        ]
    elif aux_task == 'emotions':
        # Emotion head gets 10x higher learning rate to catch up with shared encoder
        param_groups = [
            {"params": main_params, "lr": self.hparams.encoder_learning_rate},
            {"params": aux_params,  "lr": self.hparams.encoder_learning_rate * 10},
        ]
    else:
        param_groups = [{"params": self.model.parameters(), "lr": ...}]
    
    optimizer = optim.Adam(param_groups)
    
    # Linear warmup scheduler:
    # - For the first `warmup_steps` steps: LR linearly increases from 0 → encoder_learning_rate
    # - After that: LR linearly decreases from encoder_learning_rate → 0
    # This prevents large gradient updates at the start when weights are random.
    train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
    warmup_steps = int(self.hparams.warmup_proportion * train_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=train_steps,
    )
    return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
```

---

### Encoder Freezing

```python
def freeze_encoder(self):
    # Freeze all parameters in the BERT encoder (no gradient updates)
    for param in self.model.encoder.parameters():
        param.requires_grad = False
    
    # EXCEPT the last layer's output — keep that trainable
    # so the classification head has at least one layer to learn with
    for param in self.model.encoder.encoder.layer[-1].output.parameters():
        param.requires_grad = True
    
    self._frozen = True

def on_epoch_end(self):
    # After nr_frozen_epochs epochs, unfreeze the encoder
    if self.current_epoch + 1 >= self.nr_frozen_epochs:
        self.unfreeze_encoder()
```

**Why freeze?** Fine-tuning a large pre-trained model from epoch 1 can be unstable — the untrained classification head will produce random, high-magnitude gradients that corrupt the pre-trained weights. Freezing first lets the head converge, then unfreezing lets the full model fine-tune together.

---

## 4. `RoBERTaMTL.py` — The Advanced RoBERTa Architecture

This is the **more sophisticated model** with three task-specific branches instead of two.

### Key Differences from BERT version

| Feature | BERT (`RedditTransformer`) | RoBERTa (`RoBERTaMTL`) |
|---|---|---|
| Backbone | `bert-base-uncased` | `roberta-base` |
| Task branches | 2 (main + aux) | 3 (bias, emotions, social) |
| Pooling | `[CLS]` token | Mean pooling over all tokens |
| Bias head output | Raw logit | Sigmoid (0–1 directly) |
| Loss computation | In Lightning module | **Inside** `forward()` |

### Architecture Setup

```python
class RoBERTaMTL(nn.Module):
    def __init__(self, model_name="roberta-base", num_emotions=13,
                 num_social_groups=10, extra_dropout=0.0, loss_weights=None):
        
        # 1. Load base model
        base_model = AutoModel.from_pretrained(model_name, config=config)
        self.encoder = base_model
        
        # 2. Extract and deep-copy layer 12 (index 11) THREE TIMES
        layer12 = self.encoder.encoder.layer[11]
        self.layer_bias    = copy.deepcopy(layer12)  # branch for us-vs-them regression
        self.layer_emotion = copy.deepcopy(layer12)  # branch for emotion classification
        self.layer_social  = copy.deepcopy(layer12)  # branch for social group classification
        
        # 3. Truncate encoder to only run through layers 0-10 natively
        # Layers 0-10: shared, run as normal via self.encoder()
        # Layer 11: three independent copies above — we call them manually in forward()
        self.encoder.encoder.layer = nn.ModuleList(self.encoder.encoder.layer[:11])
        
        # 4. Task-specific heads
        hidden = config.hidden_size  # 768 for roberta-base
        
        self.bias_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),  # 768 → 384
            nn.Tanh(),
            nn.Dropout(...),
            nn.Linear(384, 1),               # 384 → 1
            nn.Sigmoid(),                    # squash to [0, 1] explicitly
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Dropout(...),
            nn.Linear(384, num_emotions),    # 13 emotion logits (no activation — BCEWithLogits handles it)
        )
        
        self.social_group_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Dropout(...),
            nn.Linear(384, num_social_groups),  # N class logits (CrossEntropyLoss handles softmax)
        )
```

### Mean Pooling vs CLS Pooling

```python
def _mean_pooling(self, last_hidden_state, attention_mask):
    # last_hidden_state: [B, S, 768] — one vector per token
    # attention_mask:    [B, S]      — 1 for real tokens, 0 for padding
    
    # Expand mask to match hidden state dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    # input_mask_expanded: [B, S, 768] — broadcast-expanded
    
    # Zero out padding token vectors, then sum
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)  # [B, 768]
    
    # Divide by the count of non-padding tokens (avoid div by zero with clamp)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # [B, 768]
    
    return sum_embeddings / sum_mask  # [B, 768] — average of real-token embeddings
```

**Why mean pooling for RoBERTa?** Research shows RoBERTa's `[CLS]` token isn't as well-calibrated as BERT's for downstream tasks (because RoBERTa wasn't trained with NSP — Next Sentence Prediction — which is what trains `[CLS]`). Mean pooling averages information from all tokens, often giving better sentence representations for RoBERTa.

### `forward()` with Internal Loss Computation

```python
def forward(self, input_ids, attention_mask,
            labels_bias=None, labels_emotion=None, labels_social=None):
    
    # 1. Shared layers 0-10
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    shared_hidden = outputs.last_hidden_state  # [B, S, 768]
    
    # 2. Extended mask for manual layer 12 calls
    extended_mask = self.encoder.get_extended_attention_mask(attention_mask, input_ids.shape)
    # Shape: [B, 1, 1, S] with -10000 for padding positions
    
    # 3. Three independent task-specific 12th layers
    bias_out   = self.layer_bias(shared_hidden,    attention_mask=extended_mask)
    emotion_out = self.layer_emotion(shared_hidden, attention_mask=extended_mask)
    social_out  = self.layer_social(shared_hidden,  attention_mask=extended_mask)
    
    # Each layer returns a tuple; index [0] is the hidden state tensor
    hidden_bias   = bias_out[0]   if isinstance(bias_out,   tuple) else bias_out
    hidden_emotion = emotion_out[0] if isinstance(emotion_out, tuple) else emotion_out
    hidden_social  = social_out[0]  if isinstance(social_out,  tuple) else social_out
    
    # 4. Mean pool: [B, S, 768] → [B, 768]
    pooled_bias   = self._mean_pooling(hidden_bias,   attention_mask)
    pooled_emotion = self._mean_pooling(hidden_emotion, attention_mask)
    pooled_social  = self._mean_pooling(hidden_social,  attention_mask)
    
    # 5. Task heads
    bias_score     = self.bias_head(pooled_bias)          # [B, 1], already sigmoid-ed
    emotion_logits = self.emotion_head(pooled_emotion)     # [B, 13]
    social_logits  = self.social_group_head(pooled_social) # [B, N]
    
    results = {
        "bias_score":   bias_score,
        "emotions":     emotion_logits,
        "social_group": social_logits,
    }
    
    # 6. Loss computation (only if labels are provided)
    loss = 0
    
    if labels_bias is not None:
        # MSE between predicted score and ground truth score
        loss_bias = nn.MSELoss()(bias_score.squeeze(), labels_bias)
        loss += self.loss_weights["bias"] * loss_bias
        results["loss_bias"] = loss_bias
    
    if labels_emotion is not None:
        # Binary cross-entropy for each of the 13 emotions independently
        loss_emotion = nn.BCEWithLogitsLoss()(emotion_logits, labels_emotion)
        loss += self.loss_weights["emotion"] * loss_emotion
        results["loss_emotion"] = loss_emotion
    
    if labels_social is not None and not self.ablate_social_group:
        # Cross-entropy for the social group label
        loss_social = nn.CrossEntropyLoss()(social_logits, labels_social.long())
        loss += self.loss_weights["social"] * loss_social
        results["loss_social"] = loss_social
    
    if labels_bias is not None or labels_emotion is not None:
        results["loss"] = loss
    
    return results
```

---

## 5. `RoBERTaRegression.py` — The RoBERTa Training Harness

Conceptually identical role to `BertRegression.py` but simplified and cleaner. The main difference in usage is **smart label routing**:

```python
def training_step(self, batch, batch_nb):
    inputs, targets = batch
    
    aux_task_str = str(self.hparams.aux_task)
    
    # Route the auxiliary label to the correct argument in forward()
    labels_emotion = targets.get("labels_aux") if aux_task_str == 'emotions' else None
    labels_social  = targets.get("labels_aux") if aux_task_str not in ('emotions', 'None', 'bias') else None
    
    outputs = self.forward(
        inputs,
        labels_bias=targets.get("labels"),    # always passed — it's our main task
        labels_emotion=labels_emotion,         # None if we're not doing emotions
        labels_social=labels_social            # None if we're not doing group
    )
    # The RoBERTaMTL model only computes loss for tasks where labels are provided.
    # So if labels_emotion=None, the emotion head still runs but contributes 0 to the loss.
    
    total_loss = outputs["loss"]
    self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
```

### Ablation Mode

```python
# In RoBERTaMTL:
def set_ablation_mode(self, ablate=True):
    self.ablate_social_group = ablate
    
    # Freeze the social head's parameters (no gradient updates)
    for param in self.social_group_head.parameters():
        param.requires_grad = not ablate
    
    # Also freeze the task-specific 12th layer for social
    for param in self.layer_social.parameters():
        param.requires_grad = not ablate
```

Ablation studies test whether a component is actually helping. By zeroing/freezing the social group branch, you can compare:
- Full model (bias + emotions + social group)
- Ablated model (bias + emotions only)

If performance drops significantly, social group prediction was contributing. If not, it was inert.

---

## 6. `train.py` — The Entry Point

Now that you understand all the components, the entry point is easy to read.

```python
def main(hparams):
    torch.set_float32_matmul_precision('high')
    # Ampere+ GPU optimization: uses TF32 instead of full FP32 for matrix multiplications
    # ~3x faster for matrix ops with negligible accuracy loss
    
    seed_everything(hparams.seed)
    # Sets Python, numpy, torch, and CUDA random seeds for reproducibility
    
    # Model selection
    if hparams.model_type == "roberta":
        model = RoBERTaClassifier(hparams)     # RoBERTaRegressor
    else:
        model = BERTClassifier(hparams)         # BERTRegressor
    
    # Logger: writes event files to tb_logs/<task>_<model>/version_N/
    tb_logger = TensorBoardLogger(
        save_dir="tb_logs",
        name=f"task_{hparams.aux_task}_{hparams.model_type}"
    )
    
    # EarlyStopping: if val_loss doesn't improve for `patience` epochs, stop training
    early_stop_callback = EarlyStopping(
        monitor="val_loss",   # watch this metric
        patience=hparams.patience,   # default: 10 epochs
        mode="min",           # "lower is better"
    )
    
    # ModelCheckpoint: save the model whenever val_loss improves
    ckpt_path = os.path.join(tb_logger.log_dir, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename='{epoch}-{val_loss:.2f}',   # e.g. epoch=3-val_loss=0.12.ckpt
        save_top_k=hparams.save_top_k,       # default: 1 (keep only the best)
        monitor=hparams.monitor,             # default: "val_loss"
        mode=hparams.metric_mode,            # default: "min"
    )
    
    # Save where checkpoints will live — needed by on_test_epoch_end() to write predictions.csv
    model.hparams.checkpoint_path = tb_logger.log_dir
    
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        
        accelerator="gpu" if hparams.gpus > 0 else "cpu",
        devices=hparams.gpus if hparams.gpus > 0 else 1,
        
        # For multi-GPU: 'ddp' = DistributedDataParallel (modern standard)
        # Each GPU gets a copy of the model; gradients are averaged before weight updates
        strategy="ddp" if hparams.gpus > 1 else "auto",
        
        max_epochs=hparams.max_epochs,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        # accumulate_grad_batches=2 means: do 2 forward/backward passes before one optimizer.step()
        # This effectively doubles the batch size without doubling GPU memory usage
        
        limit_val_batches=hparams.val_percent_check,   # default: 1.0 (use all validation data)
    )
    
    trainer.fit(model)    # runs the full train+val loop
    trainer.test()        # loads the best checkpoint and evaluates on the test set
```

### CLI argument parsing

```python
if __name__ == "__main__":
    parser = HyperOptArgumentParser(strategy="random_search", ...)
    
    # parser.add_argument → fixed value
    parser.add_argument("--batch_size", default=6, type=int)
    
    # parser.opt_list → can be a search space for hyperparameter tuning
    parser.opt_list(
        "--model_type",
        default="bert",
        tunable=False,      # not searched in random search mode
        options=["bert", "roberta"],
    )
    
    # Add all the BERT-specific args (encoder_model, aux_task, learning_rate, file paths, etc.)
    parser = BERTClassifier.add_model_specific_args(parser)
    hparams = parser.parse_args()
    
    if not hparams.search_mode:
        # Single run — your normal use case
        main(hparams)
    else:
        # Random search: runs main() 8 times with different sampled hyperparameters
        for hparam_trial in hparams.trials(8):
            main(hparam_trial)
```

**For your training command:**
```
python3 SentimentClassification/train.py \
    --model_type bert \          → picks BERTRegressor
    --encoder_model bert-base-uncased \  → loads 'bert-base-uncased' weights
    --batch_size 8 \             → 8 comments per GPU batch
    --gpus 1 \                   → single GPU
    --max_epochs 10              → hard limit (EarlyStopping may stop earlier)
```

All other args use their defaults:
- `--aux_task None` → no multi-task learning, pure bias regression
- `--nr_frozen_epochs 0` → fine-tune BERT end-to-end from epoch 1
- `--encoder_learning_rate 1e-5` → standard BERT fine-tuning rate
- `--train_csv Resources/UsVsThem_train_public.csv` → main training data

---

## Summary: Tensor Shapes Through the Full Pipeline

```
Input comment (raw text string)
        ↓ MyCollator / tokenizer
input_ids:       [8, 512]   — 8 comments, padded to 512 tokens each
attention_mask:  [8, 512]   — 1 for real tokens, 0 for padding
        ↓ bert.embeddings()
embedding_output:[8, 512, 768] — each token is now a 768-dim vector
        ↓ bert.encoder (layers 0-10, shared)
hidden_states:   [8, 512, 768]
        ↓ layer 11 (main copy, in BertEncoder)
hidden_main:     [8, 512, 768]
        ↓ layer 11 (aux copy, in BertEncoder)
hidden_aux:      [8, 512, 768]
        ↓ BertPooler ([CLS] token + linear + tanh)
features_main:   [8, 768]   — one sentence vector per sample, for bias task
features_aux:    [8, 768]   — one sentence vector per sample, for aux task
        ↓ classification_head (Linear 768→1)
logits_main:     [8, 1]     — one bias score per sample
        ↓ flatten + MSELoss vs labels [8]
main_loss:       scalar
```
