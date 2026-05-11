# EmotionModels Pipeline Improvements

This document summarizes the upgrades and optimizations implemented to boost the performance and robustness of the Emotion Classification pipeline.

## 1. Advanced Training: Asymmetric Loss (ASL)
We upgraded the training "learning rule" from standard Focal Loss to **Asymmetric Loss (ASL)**.
- **Why:** Emotion datasets are highly imbalanced (most emotions are "0" most of the time).
- **Impact:** ASL decouples the treatment of positive and negative samples, allowing the model to focus aggressively on difficult "zeros" without suppressing the signal from rare positive emotions.
- **Result:** Boosted the baseline Test Jaccard score from ~14% to **18.68%** (+33% relative improvement).

## 2. Threshold Optimization Pipeline
Implemented a custom post-processing pipeline (`optimize_thresholds.py`) to maximize the Macro Jaccard score.
- **Logic:** Conducts a per-class grid search (100 candidates per node) to find the perfect decision cutoff, replacing the global 0.5 default.
- **Features:** 
  - Inference caching to avoid redundant Transformer passes.
  - Calibration/Test split to prevent overfitting during threshold hunting.
  - Automatic export to `thresholds.json`.

## 3. Architecture Enhancements (`model.py`)
- **Dynamic Thresholding:** Added a `thresholds` buffer and `load_thresholds()` method to the `EmotionModel` class.
- **Path Robustness:** Updated `compute_weights_from_csv` and `RedditDataset` to automatically detect file paths regardless of whether the script is run from the project root or the sub-folder.
- **Testing Support:** Added the missing `test_dataloader` hook to enable seamless `trainer.test()` execution.

## 4. Integration & Tooling
- **Automatic Tuning:** Updated `test.py` to automatically detect and load `thresholds.json` if present.
- **Documentation:** Created and updated `explanation.md` with a detailed breakdown of the file structure, label mapping, and metrics for new team members.

## Final Recommendation
For the current **ASL-trained model (Version 4)**, the model is so well-balanced that the **default 0.5 threshold** provides the best generalization on the test set (18.68%). The threshold optimizer remains available for future models that may require custom calibration.
