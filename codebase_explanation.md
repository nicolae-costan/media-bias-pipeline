# Deep Dive: Media Bias Analysis Pipeline Explained

This technical documentation provides a granular breakdown of the `media-bias-pipeline` architecture. It is designed to serve as an academic walkthrough of our machine learning engineering decisions, model pipelines, and system configurations.

## 1. System Goals and Experimental Framework

The objective of this pipeline is to classify and interpret media bias across two distinct domains: professional journalism and user-generated social media content. To achieve this, the pipeline was split into two primary research directions:

1.  **The Baseline Experimental Track (`SentimentClassification/`):** A multi-task continuous regression model fine-tuned on Reddit comment datasets. It scores text on a continuous "Us vs. Them" scale (0.0 to 1.0) of ideological polarity.
2.  **The Relational Graph Track (`GraphNeuralNetwork/`):** A semi-supervised Graph Attention Network (GAT) trained on dense semantic embeddings and emotional probability vectors extracted from news articles.

By preserving the Reddit regression model as a **baseline comparative reference**, we demonstrate the value of transitioning from simple continuous classifications on social media comments to relational GNN structures mapped over news journalism.

---

## 2. The Toolbox: What tools are we using?
Before we dive into the code, here are the main completely free software tools the project relies on:
- **PyTorch (`torch`)**: The foundational math engine. It handles all the heavy number-crunching and "tensor" (multi-dimensional grid of numbers) math.
- **PyTorch Lightning**: A smart manager for PyTorch. Instead of writing messy code to tell the computer when to study, when to test, and how to use the Graphics Card (GPUs), Lightning handles the schedule pipeline completely automatically.
- **Hugging Face Transformers**: A library that gives us instant access to pre-trained, state-of-the-art AI brains (like BERT) without having to build them from scratch. We also use its `AutoTokenizer` to chop human words into numerical pieces.
- **Pandas and Numpy**: Tools used to read spreadsheets (CSV files) and organize massive amounts of data grids.
- **Scikit-Learn**: Used to score the AI's tests (like `jaccard_score`) and convert text labels into numbers (`LabelEncoder`).
- **Test-Tube**: Allows us to pass commands and configurations from the computer terminal easily.
- **Seaborn & Matplotlib**: Drawing tools that sketch out colorful visual graphs (heatmaps) showing where exactly the AI gets confused during tests.

---

## 3. The Brain: `RedditTransformer.py`
This file is the absolute core of the project's intelligence. It uses a state-of-the-art AI technology called **BERT (Bidirectional Encoder Representations from Transformers)**.

**What is BERT?**
Think of a standard BERT model as an office tower with 12 floors (often referred to as 12 "layers"). When a sentence enters the ground floor, it's just raw text. By the time it travels up through the 12 floors, BERT has extracted deep, abstract meaning, context, and grammar from it.

**BERT vs The Final Classification Heads**
It's important to understand the relationship between BERT and the final prediction. BERT does not spit out the final answer. Instead, BERT acts purely as an **"Encoder"** (or feature extractor). Its only job is to translate complex English text into a highly sophisticated grid of numbers representing the deep mathematical meaning of the sentence. 

These numbers are then handed off as "input data" to two totally separate, smaller Neural Networks (called "Classification Heads") sitting on the roof of the tower. These smaller networks do the actual predicting using the data BERT prepared.

**End-to-End Training (Learning as one)**
It is not a one-way street! During training, when the small neural networks on the roof make a bad guess, an error signal is sent backwards. It travels through the small networks, and keeps going backward all the way down through BERT's 12 floors. This means BERT is physically being re-wired and fine-tuned to understand Reddit-specific context at the exact same time the final neural networks are learning how to output the final score. They learn together as one giant, unified brain.

### The Clever Modification in this Project
Normally, you would just take the final output from the 12th floor and use it to guess the answer. But because we are doing *Multi-Task Learning* (answering two questions at once), `RedditTransformer.py` does something brilliant:

**It splits the top floor into two separate rooms.**
- Floors 1 through 11 are **shared**. Both the main task (Us vs. Them) and the auxiliary task (Emotions) use the exact same floors. This makes sense because basic English language rules (like "what is a noun" and "what is a verb") don't change regardless of what question you are trying to answer.
- Floor 12 is **split**. We create `layer_main` and `layer_aux`. 
  - `layer_main` receives info from floor 11 and specializes exclusively in figuring out the "Us vs. Them" score.
  - `layer_aux` receives the exact same info from floor 11 but specializes exclusively in figuring out the emotions.

### Breaking down the Code in `RedditTransformer.py`:
1. **`class RedditTransformer:`**
   This is the main structural wrapper. It loads a pre-trained BERT model built by Google/Hugging Face. If we tell the code that we have a secondary task (`num_groups is not None`), it swaps out standard top floors for our custom split components (`BertEncoder` and `BertPooler`). It also manages the **Attention Mask** — a filter of 1s and 0s that tells the AI to completely ignore the blank, padded space attached to the end of short sentences.
2. **`class BertEncoder:`**
   - In the setup, it takes the 12 standard layers: `self.layer = torch.nn.ModuleList(layers[:-1])`. The `[:-1]` part is python code meaning "take everything EXCEPT the very last item" (so floors 1 to 11).
   - Then it grabs the final 12th layer and makes a completely separate clone of it: `layer_main` and `layer_aux`.
   - In the `forward` function (which is the actual pathway the data travels), it passes the text through the 11 shared layers. Then, it sends the results into *both* the `layer_main` and `layer_aux` independently. It outputs two different sets of numbers for the pooler layer.
3. **`class BertPooler:`**
   Instead of just looking at the very first word (`[CLS]`) as a summary, the AI now scans the entire sentence and uses **Mean Pooling**. It adds up the emotional weight of every single real word (while completely ignoring blank padding spaces) and calculates the average. This is crucial because emotions are often hidden in single energetic words anywhere in the text (like "furious" at the end of a sentence). It then passes this through a modern mathematical filter called `GELU` (which lets strong emotions flow through easily without getting stuck), uniquely separated for both the main and auxiliary tasks.

---

## 4. The Engine: `BertRegression.py`
While `RedditTransformer` is the model's physical neural brain, `BERTRegressor` is the engine that drives how it learns over time. It is built using **PyTorch Lightning**, a framework that organizes messy AI math into clean, standard, readable steps.

### What does it do?
1. **The `forward` function:**
   This is the front door. Raw data comes in, gets handed to the `RedditTransformer`, and the two predictions (Main score and Emotion score) pop out.
2. **The `loss` function (How the AI learns from its mistakes):**
   When the AI makes a guess, it compares its guess to the real, correct answer (which a human expert provided in the data). The mathematical difference between the AI's guess and the truth is called **Loss**.
   - For the main task (predicting a sliding number scale), it checks loss using `MSELoss` (Mean Squared Error).
   - For the auxiliary task (Categorizing multiple specific emotions or a bias group), it uses `CrossEntropyLoss` or `BCEWithLogitsLoss`. For emotions, it now uses a smart **weight scaling system** that forces the AI to put a lot more penalty on getting rare emotions wrong, otherwise, the AI gets lazy and just predicts 0 for everything rare.
3. **Freezing / Unfreezing and Optimizers:**
   Because BERT is already very smart out-of-the-box (pre-trained on billions of words by Google), we don't want our new tasks to destroy its foundational knowledge. 
   - We **freeze** the lower floors temporarily (meaning we physically lock them from being changed) so early mistakes don't ruin them.
   - We use different **learning rates**. The old, smart layers learn very slowly (a tiny learning rate), while the newly added custom top floors learn faster (about 3 times as fast to stop it from memorizing the data). We also use **Warmup Steps**—meaning the engine slowly warms up before it starts changing the brain aggressively.
4. **The `backward` function & GradNorm (The Secret Sauce):**
   Here is a major problem with making an AI do two things at once: If one task is really easy and the other is really hard, the AI gets lazy. It will focus entirely on getting the easy task 100% correct and ignore the hard one. 
   To prevent this, the code uses a complex mathematical technique called **GradNorm** (Gradient Normalization). 
   - It calculates how fast the AI is "learning" on the main task versus the auxiliary task.
   - It gives each task a **weight multiplier** (`self.weights`).
   - If one task is learning too fast, it dynamically shrinks its weight multiplier, forcing the AI to slow down and focus on the struggling task instead.
5. **Testing and Validation (`validation_step` & `test_step`):**
   After practicing for a bit, the AI takes a break and tests itself on data it has never seen (the Validation set). During this test, it calculates accuracy and creates visual tools:
   - **Confusion Matrices**: Heatmap grids uploaded to TensorBoard showing exactly where the AI got confused (e.g. if it constantly mistakes 'Anger' for 'Disgust').
   - **Pearson Correlation**: A statistical measure of how closely its numerical guesses actually match the true upward/downward trends of the human "Us vs Them" bias scale.

---

## 5. The Translator: `dataloader.py`
AI cannot read English words like "banana" or "government"; it strictly only understands math and numbers. The purpose of `dataloader.py` is to translate human text into structured numbers.

1. **`RedditDataset`**: This opens and reads the CSV spreadsheets containing the raw Reddit comments and loads them into memory. If we're tracking emotions, it specifically filters down to 7 strong feelings (like Anger or Sympathy) and ignores rare ones (like Relief) because there aren't enough examples for the AI to learn from. It also uses the Scikit-Learn `LabelEncoder` to perfectly map text descriptions into numeric classes. **Label Consistency Rule:** The code ensures it only strictly *translates* categories based on what it mapped out in the very beginning, preventing it from randomly scrambling the translation dictionary between Training and Testing sets.
2. **`MyCollator`**: This does the exact cleaning and translation. Specifically it runs `__call__` for each batch of data:
   - **Regex Link Cleaning**: Whenever it sees a web URL in a comment (like `http://website.com`), it deletes it and replaces it with the plain word `LINK`. URLs usually confuse AI because they are random strings of chaotic text.
   - **Tokenizer**: It uses an `AutoTokenizer` provided by Hugging Face. This chops the sentences into smaller puzzle pieces (called tokens) and converts them into numeric IDs that BERT understands (e.g., the word "apple" might be token ID #4592). It also generates the aforementioned attention masks.
   - **Padding**: AI requires every sentence submitted to be the exact same length (e.g., exactly 512 tokens long). If a sentence is shorter, the collator adds "padding" (blank zeros) to the end. If it's too long, it cuts off the rest to keep it strictly uniform.

---

## 6. The Drivers: `train.py` and `test.py`
These are the execution scripts. They are what you run from your computer terminal to unleash everything built above.

### `train.py`
1. **Arguments (`HyperOptArgumentParser`)**: This sets up all the configurable knobs and dials for your experiment. E.g., `batch_size` (how many sentences the AI should study simultaneously), `learning_rate` (how drastically to alter its brain based on mistakes), and `max_epochs` (how many times the AI should re-read the entire dataset from start to finish).
2. **Modern Hardware & Software Initialization**: It explicitly commands PyTorch to boot into "high-precision matrix math", empowering newer Graphics Cards (specifically Ampere series or newer) to calculate tasks much faster. It uses PyTorch Lightning's native `seed_everything` for reliable randomness. It also injects a PyTorch safety fix (`add_safe_globals`) enabling it to safely unpackage files from disk without security crashes, and strictly enforces building the engine with BERT—blocking unsupported "RoBERTa" variations.
3. **Callbacks (The AI Supervisors)**:
   - **EarlyStopping**: It watches the AI take its tests. If the AI completely stops improving after a few tries (a number defined by `patience`), it halts the training early so we don't waste hours of time and electricity.
   - **ModelCheckpoint**: Every time the AI gets a new high score on a validation test, it automatically saves a copy of its physical "brain state" to a `.ckpt` (checkpoint) file.
4. **Trainer**: Boots up PyTorch Lightning's `Trainer` which seizes control of the computer's Graphics Card (GPU) resources using a Distributed Data Parallel (DDP) strategy, then runs the `trainer.fit()` loop to officially begin the learning process.

### `test.py`
1. This is used when training is completely finished. Instead of training, it scans the logs folder looking for the `.ckpt` file (the saved brain from training).
2. It loads that exact brain state using `BERTRegressor.load_from_checkpoint`.
3. It calls `trainer.test()`, which passes the final, unseen, true Test Dataset through the brain one last time to measure how good it really is in the real world. 
4. It then spits out the final metrics and structurally saves all of its outputs into a `predictions.csv` file so humans can review where it was right and wrong.

---
### Summary of the Lifecycle
1. The raw Reddit data gets cleaned and turned into math (`dataloader.py`).
2. Training pre-settings, high-speed optimizations, and GPUs are configured (`train.py`).
3. The brain (`RedditTransformer.py`) looks at a sentence, processes it through 11 shared layers, splits down two isolated paths at the 12th layer, and spits out two answers (Us vs. Them scale AND Emotion).
4. The Engine (`BertRegression.py`) reviews those two answers against the human labels, realizes how mathematically "wrong" it is, dynamically balances the focus between the two tasks using GradNorm, and sends a signal backwards to adjust its brain pathways.
5. Millions of sentences later, the AI gets highly accurate. Once it's finished, `test.py` proves its accuracy against completely hidden test data and writes out its findings to the CSV file.
