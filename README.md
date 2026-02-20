# Spam-Classification-Model
This project builds a spam message classifier that automatically separates spam from normal messages. It learns patterns based on the types of words used and how frequently they appear in each category. The same approach can later be applied to organize emails, research notes, reports, or other text-based data.
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
