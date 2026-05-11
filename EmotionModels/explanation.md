# THIS IS FOR DUMBASSES WHO DON'T UNDERSTAND THE CODE (like rares) 


# EmotionModels Pipeline Explanation

The `EmotionModels` directory is a streamlined PyTorch Lightning pipeline explicitly built to fine-tune a pre-trained Transformer model (by default, `SamLowe/roberta-base-go_emotions`) on the dataset's 13 specific emotion labels. 

Unlike the main `SentimentClassification` pipeline which builds complex custom layers for multi-tasking, this pipeline is much more direct and focuses heavily on handling the severe class imbalance typical in multi-label emotion data.

Here is a file-by-file breakdown of what is going on:

### 1. `dataloader.py`
This file is responsible for reading the CSVs and feeding batches of data into the model.
*   **The Dataset (`RedditDataset`)**: It isolates the text column (`body`) and the 13 emotion columns (Anger, Contempt, Disgust, Fear, etc.). It stores these labels efficiently as a `numpy` float32 array instead of a list, making it highly optimized for parallel data loading.
*   **The Collator (`MyCollator`)**: When the data is batched, this collator first cleans the text by replacing all URLs with the literal string `'LINK'`. It then tokenizes the text using HuggingFace's `AutoTokenizer` and stacks the labels into a `[BatchSize, 13]` PyTorch tensor.

### 2. `model.py`
This is the heart of the pipeline. It contains the neural network architecture, loss calculation, and optimization logic wrapped in PyTorch Lightning's `LightningModule`.
*   **The Model Backbone**: Instead of building a custom Transformer from scratch, it leverages `AutoModelForSequenceClassification`. It defaults to the `SamLowe/roberta-base-go_emotions` model, which was already trained to recognize 28 different emotions on a massive Reddit dataset.
*   **The Label Mapping Trick**: The pre-trained model outputs 28 emotions, but this project's dataset only has 13 labels. Rather than throwing away the pre-trained classification head, the code maps the 13 custom labels onto the 28 GoEmotions nodes. For example, the project's "Anger" label maps to the "Anger" node in GoEmotions, "Contempt" maps to "Disapproval", etc.
*   **Handling Class Imbalance (`compute_weights_from_csv` & `FocalLoss`)**: Emotion datasets are notoriously imbalanced (e.g., "Neutral" is very common, "Guilt" is very rare). 
    *   The model dynamically reads the CSVs before training to calculate how rare each label is and assigns higher gradient weights to rare labels.
    *   It uses a custom **Binary Focal Loss** (`FocalLoss`). Standard cross-entropy treats all errors equally. Focal Loss mathematically forces the model to focus its learning capacity on "hard" examples (where it is currently predicting poorly) and ignores easy, well-predicted examples.
*   **Metrics**: It logs the `val_loss` and the `jaccard_score` (a standard metric for multi-label classification measuring the intersection over union of predicted vs true labels).

### 3. `train.py`
This is the execution script to start training.
*   It sets up the PyTorch Lightning `Trainer`.
*   It configures a `TensorBoardLogger` to save logs to a `tb_logs/emotion_classification` folder.
*   It sets up two Callbacks: 
    *   **EarlyStopping**: Stops training if the validation loss doesn't improve for a certain number of epochs (patience).
    *   **ModelCheckpoint**: Saves the best `.ckpt` model state automatically.
*   It calls `trainer.fit()` to train, and immediately calls `trainer.test()` when training concludes to evaluate the best model on the test set.

### 4. `test.py`
A simple utility script used strictly for evaluating an already-trained model.
*   It takes a `--checkpoint_path` argument.
*   It searches that folder for a `.ckpt` file, loads the `EmotionModel` from those weights, and spins up a `Trainer` in test-only mode (forced to run on CPU).
*   It runs `trainer.test()` to output the final Test Jaccard Score and Loss.

---

## Key Concepts (FAQ)

### What is a "Transformer"?
A Transformer is a type of neural network architecture invented by Google in 2017. It is the underlying technology behind modern AI like ChatGPT, BERT, and RoBERTa. Instead of reading text word-by-word like older models, a Transformer looks at the entire sentence at once using a mechanism called "Self-Attention." This allows it to deeply understand the context and relationships between words (e.g., understanding that "bank" means a river bank vs. a financial bank based on the surrounding words).

### What is `SamLowe/roberta-base-go_emotions` and where does it come from?
It comes from **Hugging Face**, which is essentially the GitHub for machine learning models. 
*   `SamLowe` is the user who uploaded it.
*   `roberta-base` is the foundational model (an optimized version of BERT created by Facebook).
*   `go_emotions` means this specific model was already trained on a famous Google dataset called "GoEmotions", which categorized text into 28 different emotions.
When the code runs `AutoModelForSequenceClassification.from_pretrained(...)`, it automatically downloads this pre-trained brain from the internet.

### Is the code actually training a model, or just mapping labels?
**It is actually training!** Here is exactly what happens:
1.  **The Starting Point:** We download the `SamLowe` model. This model already has a good general understanding of English and knows how to recognize 28 emotions.
2.  **The Mapping:** Your dataset only has 13 emotions. The mapping trick just tells the code *where to look*. For example, it says, "When checking our 'Anger' label, look at the model's 'Anger' output node."
3.  **The Actual Training (Fine-Tuning):** When you run `train.py`, the model reads your political Reddit comments and makes predictions. Because the language in Reddit debates is very different from general text, the model will initially make mistakes. The code calculates this error (using the `FocalLoss`) and uses it to **update the model's internal weights**. So, the model is actively learning and adapting its "brain" to specifically understand emotions within the context of your Us-vs-Them bias dataset.

### What is a "Threshold"?
Because this is a **multi-label** problem (a comment can have multiple emotions at once), the model doesn't just pick the #1 top emotion. Instead, it looks at *all 13 emotions* and assigns a confidence score to each one (e.g., "I am 80% sure this is Anger, 60% sure it's Disgust, and 10% sure it's Joy").
The **Threshold** is the cutoff point where you decide, "Yes, this emotion is officially present." 
In your code, the threshold is implicitly set at 50% (you can see this in `model.py` where it does `preds_28 = (logits_28 > 0).float()`). If the model is more than 50% confident that an emotion is there, it flags it as a `1` (True). If it's less than 50% confident, it flags it as a `0` (False).

### What is the "Jaccard Score"?
The Jaccard Score (also known as Intersection over Union) is a grading system used to see how well the model is performing, specifically designed for multi-label tasks. 
Imagine a comment's **true labels** are: `[Anger, Contempt]`.
The model's **predictions** are: `[Anger, Disgust]`.

The Jaccard score asks: *Out of all the unique emotions involved, how many did they agree on?*
*   **Intersection (What they agreed on):** `[Anger]` (1 emotion)
*   **Union (All unique emotions involved):** `[Anger, Contempt, Disgust]` (3 emotions)
*   **Jaccard Score:** 1 / 3 = **0.33** (or 33%)

A score of 1.0 (100%) means the model perfectly guessed every single emotion and didn't guess any extra ones. It's a very strict, but very accurate way to measure success when multiple answers can be right at the same time.

### How to find the "Best" model?
Since you have 11+ versions, you might be wondering which one actually learned the most. Here is how you can tell:

1.  **Look at the filename:** The checkpoints are named like `epoch=4-val_loss=0.1060.ckpt`. 
    *   The `val_loss` is the "Error Score." 
    *   **LOWER is better.** (0.1060 is much better than 0.2167).
2.  **Look at the Epoch:** `epoch=8` means the model studied the entire dataset 9 times. Usually, models with more epochs have had more time to learn.
3.  **Check for empty folders:** Many of your 11 versions might be empty if the training was stopped early. If there is no `checkpoints` folder inside a version, that run didn't finish.

**Currently, your best model is in:**
`EmotionModels/tb_logs/emotion_classification/version_6/checkpoints/epoch=4-val_loss=0.1060.ckpt`
*(This one has the lowest error score of all your saved models).*
