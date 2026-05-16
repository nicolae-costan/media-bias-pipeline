# Multi-Task Media Bias & Sentiment Classification: Detailed Developer Guide

This document dives deeply into the specifics of this multi-task learning pipeline. It covers the core algorithmic objectives, dependencies, how the modules connect, explains important Deep Learning concepts (like `[CLS]`), and provides detailed, function-by-function explanations.

---

## 1. Project Objective and Context

The primary architectural goal of this codebase is to quantify polarizing **"Us vs. Them" bias** within raw social media text (e.g., Reddit comments) along a numerical scale. 

To achieve state-of-the-art accuracy, this network leverages **Multi-Task Learning (MTL)**. Instead of solely predicting the "Us vs. Them" scale directly from text features, the model is simultaneously penalized during backpropagation for failing a secondary target prediction (the auxiliary task). The auxiliary tasks configured here include:
1. Multi-label emotion classification (scoring anger, contempt, disgust, happiness, etc.)
2. Categorical group classification (identifying the target demographic).

By propagating mathematical gradients back through the network from *both* the primary continuous regression objective and the secondary classification objective simultaneously, the shared hidden layers of the transformer are forced to learn deeper, more abstract semantic correlations.

---

## 2. Core Libraries and Dependencies

The project relies on an ecosystem of data processing and deep learning libraries. Here is what each one does in the context of this pipeline:

*   **`torch` / `torch.nn` (PyTorch)**: The foundational tensor and neural network library. Used for defining custom network layers, loss functions (MSE, CrossEntropy), and performing automatic differentiation (gradient computation).
*   **`pytorch_lightning` (PL)**: A high-level wrapper over PyTorch. It removes the boilerplate of writing custom training loops (`for epoch... for batch...`). It automatically handles distributed training across GPUs, moving data to devices, accumulating gradients, and validation loops.
*   **`transformers` (Hugging Face)**: Used to retrieve state-of-the-art pre-trained language models (like BERT). 
    *   *`AutoModel` / `AutoConfig`*: Used to dynamically load model architectures and their configurations.
    *   *`AutoTokenizer`*: Converts raw strings into token IDs and attention masks expected by the transformer.
*   **`pandas` / `numpy`**: Used heavily in `dataloader.py` and `BertRegression.py` for reading datasets from CSV files, manipulating structures, concatenating sets, and handling multi-dimensional computational arrays (e.g., argmax predictions).
*   **`scikit-learn` (`sklearn`)**: Used for traditional machine learning operations:
    *   *`LabelEncoder`*: Converts categorical string labels (e.g., specific bias tags) into numerical IDs.
    *   *`jaccard_score` & `confusion_matrix`*: Metrics for evaluating classification splits.
*   **`test_tube` (`HyperOptArgumentParser`)**: Extends Python's default `argparse` to simplify running multi-configuration searches (hyperparameter sweeps) via terminal execution.
*   **`seaborn` & `matplotlib`**: Visualization libraries used during validation/testing to render Confusion Matrices as images that are then uploaded to TensorBoard.

---

## 3. System Flow: How the Files Connect

The system is designed hierarchically. Here's exactly what happens and how data flows when you run a training session:

1. **Entry Point** (`train.py` or `test.py`): You execute these scripts from the terminal. They parse your arguments (like learning rate, batch size) and set up the physical environment via the PyTorch Lightning `Trainer` object. 
2. **Model Instantiation**: `train.py` explicitly creates an instance of the `BERTRegressor` class (found in `BertRegression.py`).
3. **Architecture Mapping**: When `BERTRegressor.__init__()` executes, it bootstraps the components it needs:
   - It sets up data processing pipelines using definitions from `dataloader.py` (specifically `MyCollator` and `RedditDataset`). 
   - It builds the physical neural network by importing and instantiating `RedditTransformer` (from `RedditTransformer.py`).
4. **Execution Loop**: The Lightning `Trainer` object takes control (via `trainer.fit()`). Behind the scenes, the Trainer loops automatically: it calls `train_dataloader()` to fetch data arrays, passes those batches into `training_step()`, which executes the network's `forward()` pass, computes the `loss()`, and then runs `optimizer_step()`.

---

## 4. Key Transformer Concepts (Terminology)

Before understanding the functions, let's clarify key terms heavily referenced in the code:

- **The `[CLS]` Token**: In BERT and similar models, a special `[CLS]` (Classification) token is automatically prepended to the start of every text sequence. While standard models isolate this single token for classification (because it accumulates global context), this project now departs from that logic for emotion detection. Instead of using `[CLS]`, it currently implements **Mean Pooling**.
- **Attention Mask**: Because text inputs are padded with zeroes to guarantee batches are perfectly rectangular matrix shapes, the `attention_mask` is an array of `1`s (real words) and `0`s (padding). This project manually extends this into a 4-D structure to properly bypass standard BERT forwarding processes without losing mask dimensions.
- **Pooler**: The neural component that extracts the textual features and applies a dense neural network layer to shape it into a final vector. Previously this isolated `[CLS]` and applied `Tanh`, but it has been upgraded to properly **Mean Pool** only non-padded tokens and uses `GELU` (Gaussian Error Linear Unit). `GELU` avoids the saturation limit of `Tanh`, allowing dynamic emotional features to flow effectively into the classification head without compression.
- **Encoder vs. Classification Heads**: BERT acts exclusively as a contextual feature extractor (the Encoder). It maps raw token inputs into dense vector representations. These representations are then fed as "input data" into disjoint, comparatively shallow Multi-Layer Perceptrons (MLPs) known as "Classification Heads" which perform the actual logit prediction. 
- **End-to-End Fine-Tuning**: The gradient errors computed at the classification heads (MSE or BCE) are backpropagated entirely through the network. This means the pre-trained BERT weights are not frozen linearly (except during explicit warmup phases); they are dynamically fine-tuned end-to-end to capture domain-specific Reddit semantics alongside the classification heads.

---

## 5. Detailed File-by-File Code Breakdown

### `dataloader.py`
*Handles reading data from disk and transforming text into network-ready numeric tensor format.*

*   **`sentiment_analysis_dataset(hparams, train=True, ...)` Function**: 
    *   **How it's used**: Factory function called inside `BertRegression.py` when standard PyTorch Lightning data loading hooks (`train_dataloader()`, `val_dataloader()`) execute.
    *   **What it does**: Instantiates and returns a `RedditDataset` matching the specific mode requested (train/test/validation).
*   **`RedditDataset(Dataset)` Class**:
    *   **How it's used**: Wrapped inside a PyTorch `DataLoader` to represent the list of total records.
    *   **`__init__(data_csv, aux_task, le, le_aux)`**: Reads the loaded CSV into memory. If the auxiliary task is `"emotions"`, it explicitly squashes **only the 7 most prominent emotion columns** (discarding rare ones like Relief or Happiness that lack sufficient training data) from the CSV into one continuous PyTorch list (for multilabel classification). If the task is explicitly `'None'`, it hardcodes labels to zero. Otherwise, it applies the `LabelEncoder` (`le_aux`) to change textual categorical tags into numerical integers. *Crucially, it utilizes `.transform()` rather than `.fit_transform()`: this intentionally preserves the global index mappings between train, test, and dev sets without inadvertently scrambling category definitions when loading new dataset fractions.*
    *   **`__getitem__(idx)`**: Standard PyTorch requirement. Called sequentially when the `DataLoader` iterates. It fetches a strict row index from the dataset, extracting the comment body, its bias labels, and its scale metric.
*   **`MyCollator(object)` Class**:
    *   **How it's used**: Passed as the `collate_fn` argument to the PyTorch `DataLoader` inside `BertRegression.py`. A collator directs the loader on how to correctly merge multiple disparate rows from `RedditDataset` into one uniform "batch" tensor.
    *   **`__call__(batch)`**: Loops over the rows fetched by `__getitem__`. It specifically utilizes Regex (`re.sub(...)`) to replace complex web URLs with the uniform string `"LINK"`, because unique URLs fragment the tokenizers. It then calls Hugging Face's `AutoTokenizer` on the batch to convert vocabulary words into numerical `input_ids` and `attention_mask`s integers, zero-padding them uniformly to `self.max_length`.

### `RedditTransformer.py`
*The intricate deep-learning wiring. This isn't just an off-the-shelf model. It intercepts the internal circuitry of a standard BERT model to create "split-ends" for Multi-Task Learning.*

*   **`RedditTransformer(torch.nn.Module)` Class**: 
    *   **`__init__(... )`**: Scaffolds the standard Hugging Face `AutoModel`. If the user passes an auxiliary task, it physically deletes the default encoder block (`self.encoder.encoder = BertEncoder(...)`) and pooler (`self.encoder.pooler = BertPooler(...)`), overriding them with custom Multi-Task variants below.
    *   **`forward(batch)`**: This acts as the physical entry point to the tensor path. It receives `input_ids` and `attention_mask` tensors. In standard BERT, `AutoModel.forward()` handles everything, but to force data through the custom split layers, this function violently bypasses normal operations: it calculates the uncontextualized token embeddings itself, forces a "4-D extended attention mask", and feeds tokens directly into our custom `self.encoder.encoder`.
*   **`BertEncoder(torch.nn.Module)` Class**: 
    *   **How it's used**: Acts as a substitute for BERT's native transformer blocks.
    *   **`__init__(config, layers)`**: Slices the original 12 pre-trained transformer layers. It keeps layers 1 to $N-1$ strictly shared (meaning both tasks learn from a shared representation). However, it clones the final $N$-th layer into two completely disjoint matrices: `layer_main` and `layer_aux`.
    *   **`forward(...)`**: Iterates data through a `for` loop over the shared layers. Upon reaching the final sequence block, it channels the data once through `layer_main` and parallelly through `layer_aux`, creating two uniquely specialized neural arrays representing the two different tasks.
*   **`BertPooler(torch.nn.Module)` Class**: 
    *   **`forward(hidden_states)`**: Rather than extracting the `[CLS]` token, this uses **Mean Pooling** over non-padding tokens. The assumption is that emotion signals are often scattered across single words (e.g. 'devastated') rather than pre-trained into the NSP-focused `[CLS]` token. It computes the mathematical block mean for sequence indices mapped to the active `attention_mask`, passes the aggregates independently through split `dense_main` and `dense_aux` layers, and applies `GELU` activation to preserve variance.

### `BertRegression.py`
*The orchestration module. Maps incoming data arrays to network layers and establishes loss computation optimization logic.*

*   **`BERTRegressor(pl.LightningModule)` Class**:
    *   **`__init__(hparams)`**: Takes the hyperparameter configurations, sanitizes out internal `test_tube` dictionary namespace parameters to ensure TensorBoard saves correctly, registers `MyCollator` onto the class, and instantiates the `RedditTransformer` class mapped to `self.model`.
    *   **`forward(batch)`**: Wraps Lightning module definitions. Takes the dictionary output processed uniformly from `MyCollator`, feeds it to `self.model`, and catches `logits_main`, `logits_aux` out the other side.
    *   **`loss(predictions, targets)`**: Automatically checks configurations. Uses `MSELoss` (Mean Squared Error) because the primary target (`usVSthem_scale`) is a continuous number. If secondary targets are multi-label (`emotions`), applies `BCEWithLogitsLoss` infused with calculated `pos_weight`. By statically computing the imbalance penalty `neg_count / pos_count` directly from the training dataframe, it heavily penalizes failures on rare classes preventing the network from settling on zero-confidence thresholds to cheat the loss metric. 
    *   **`backward(loss)`**: Ordinarily, calling `.backward()` is a 1-line process in standard PyTorch, but this overrides it to implement an adaptive mathematical technique called **GradNorm**. It tracks the backward gradients (partial derivatives) of the specific last transformer layer. If the auxiliary task generates huge gradients compared to the main task, this mathematical logic dynamically scales down the aux task's backward momentum (`self.weights`). This stops the secondary task from destroying features critical to the main objective (catastrophic forgetting). 
    *   **`training_step(batch, batch_nb)`**: Defines an atomic stage-loop step. Receives batches from `train_dataloader`, feeds them to `self.forward()`, calculates `self.loss()`, multiplies the loss magnitudes by the dynamic `self.weights`, and tells PyTorch Lightning to execute backpropagation updating the weights.
    *   **`on_validation_epoch_end(...)`**: Gathers all evaluation tensors post-loop natively handling DDP logic. Crucially, it computes the statistical **Pearson Correlation** array (since pure categorical accuracy is insufficient for judging scale regressions). For emotion tasks, the `Jaccard_score` is aggregated safely over the *full* epoch logic avoiding zero-division issues introduced by batch fragmentation, leveraging seaborn and Matplotlib to sketch actual colored Heatmap images mapping out `confusion_matrix` grids.
    *   **`configure_optimizers()`**: Creates an `Adam` optimizer block alongside "Warmup Steps" (meaning the multiplier starts at zero and gradually ramps up—vital to avoid detonating pre-trained Transformer weights instantly!). Employs a split learning rate logic: the pre-trained `encoder_learning_rate` gets an astronomically tiny learning factor (`1e-5`), while newly initialized custom auxiliary emotion heads train faster (3x the base speed) avoiding extreme initial overfitting compared to historical 10x models. Also utilizes `freeze_encoder()` logic to stop the base layers from being altered in the very early epochs.

### `train.py` & `test.py`
*The hardware execution distribution layers. They never touch tensor math directly; they parse settings and define hardware deployment instructions.*
*   **`train.py` -> `main()`**: It invokes PyTorch Lightning's `Trainer()` class. 
    * **Modern PyTorch & Hardware Optimizations**: It triggers `torch.set_float32_matmul_precision('high')` locally optimize calculation arrays for Ampere+ GPU architectures (Tensor Cores). Because `train.py` loads `scikit-learn` modules locally during testing, it registers PyTorch 2.6+ unpickling safety bypasses (`torch.serialization.add_safe_globals([LabelEncoder])`) inherently blocking unpickling-error-crashes previously associated with safe state loading. Finally, code execution forces all requests aimed at `roberta-base` encoders firmly back into valid `bert-base-uncased` constraints to sustain compatibility.
    * **Distribution Mechanism**: Sets up parameter allocations (`accelerator="gpu"`, `"devices"` mapping), dictates `ddp` (Distributed Data Parallel) algorithms to handle massive memory footprints efficiently, and attaches monitoring callbacks (`TensorBoardLogger()`, `ModelCheckpoint()`).
*   **`test.py` -> `main()`**: Operates differently. Instead of initializing models from scratch arrays, this runs a localized folder scan locating the historical most-optimal `.ckpt` configuration parameters inside `tb_logs`. It unwraps those frozen weights using `load_from_checkpoint()`, then injects that strict snapshot into `trainer.test()` allowing `BertRegression` to execute `test_step()` benchmarks against strictly unseen external data caches.

---

## 6. Execution Lifecycle Summary

1. Data instances stream continuously from disk via `RedditDataset` through the `MyCollator`, receiving on-the-fly Regex cleanup and tokenization into padded multi-dimensional integer tensors (`dataloader.py`).
2. High-level orchestrators prepare DDP environments mapping to hardware accelerators, instantiating model checkpoints and random seeds (`train.py`).
3. Tensors enter the Transformer architecture (`RedditTransformer.py`), traversing $N-1$ shared multi-head attention blocks. At the final $N$-th layer, operations explicitly fork, outputting two distinct sets of `[CLS]` sequence aggregates via customized poolers.
4. The custom wrapper module (`BertRegression.py`) computes MSE loss against the regression scale and BCE/CrossEntropy loss against the aux targets. 
5. During the `backward()` pass, the `GradNorm` algorithm evaluates the partial derivative norms of the shared representation. It scales learning gradients inversely to their relative magnitudes, preventing the network from catastrophically ignoring challenging tasks.
6. Checkpointing mechanisms autonomously preserve optimized parameter states to `.ckpt` binaries upon observing minimized validation losses. 
7. Finally, offline batch scoring (`test.py`) restores parameter states, runs pure-inference forward passes on unseen evaluation sets, and dumps the final scalar estimates into `predictions.csv`.
