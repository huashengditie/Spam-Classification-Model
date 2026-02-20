# Spam-Classification-Model
A machine-learning spam classifier that predicts spam vs non-spam by analyzing word choice and word-frequency patterns in text messages.
Implemented an end-to-end pipeline including dataset preprocessing, CountVectorizer feature extraction, Multinomial Naive Bayes training, and confusion matrix evaluation.
Simplified version of same language-pattern approach to healthcare research, such as tracking patient recovery progress by analyzing changes in word usage and emotional tone in clinical notes or patient journaling.
"""
README

### What this does
- Trains a Multinomial Naive Bayes classifier on the SMS Spam Collection dataset.
- Classifies sample messages as SPAM or HAM.
- Saves four charts:
  - spam_vs_normal_donut.png
  - normal_top_terms.png
  - spam_top_terms.png
  - classification_confusion_matrix.png

### How to run
1) Ensure dependencies are installed:
   pandas, numpy, scikit-learn, matplotlib, seaborn
2) Place spam.csv in the same folder.
3) Run:
   python mainn.py

### Outputs
- Prints predictions for example messages.
- Writes PNG files to the project directory.

### Notes
- The word-frequency charts use hardcoded sample term counts for display.
- The confusion matrix uses a pink macaron color gradient.
"""
