import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


CONFIG = {
    "data_file": "spam.csv",
    "label_map": {"ham": 0, "spam": 1},
    "pie_output": "spam_vs_normal_donut.png",
    "ham_output": "normal_top_terms.png",
    "spam_output": "spam_top_terms.png",
    "cm_output": "classification_confusion_matrix.png",
    "font_stack": ["Comic Sans MS", "DejaVu Sans"],
    "pink_pair": ["#F8BBD0", "#F48FB1"],
    "pink_cmap": ["#FFF5F8", "#FADDE7", "#F6B7D2", "#F18BB9", "#E8589A"],
}


def load_sms_dataset(path):
    df = pd.read_csv(
        path,
        encoding="latin1",
        usecols=[0, 1],
        names=["Label", "Text"],
        header=0
    )
    df["is_spam"] = df["Label"].map(CONFIG["label_map"])
    return df


def train_classifier(frame):
    x_train, x_test, y_train, y_test = train_test_split(
        frame["Text"],
        frame["is_spam"],
        random_state=18,
        test_size=0.3
    )
    vectorizer = CountVectorizer()
    x_train_bow = vectorizer.fit_transform(x_train.values)
    model = MultinomialNB()
    model.fit(x_train_bow, y_train)
    return model, vectorizer, x_test, y_test


def classify_messages(msg_list, model, vectorizer):
    msg_bow = vectorizer.transform(msg_list)
    preds = model.predict(msg_bow)
    return ["SPAM" if p == 1 else "HAM" for p in preds]


def annotate_bars(bar_objs):
    for rect in bar_objs:
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 2,
            height + 2,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9
        )


def render_charts():
    plt.rcParams["font.family"] = CONFIG["font_stack"]
    rose = CONFIG["pink_pair"]

    class_names = ["Normal", "Spam"]
    class_counts = [4827, 747]
    labels_with_n = [f"{name}: {n}" for name, n in zip(class_names, class_counts)]
    plt.pie(
        class_counts,
        labels=labels_with_n,
        colors=rose,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white"}
    )
    plt.title("Spam vs Normal Message Counts")
    plt.axis("equal")
    plt.savefig(CONFIG["pie_output"])
    plt.close()

    top_ham_terms = {"meeting": 100, "today": 80, "tickets": 60, "movie": 40, "coffee": 20}
    plt.figure(figsize=(8, 4))
    idx = np.arange(len(top_ham_terms))
    ham_bars = plt.bar(idx, top_ham_terms.values(), color=rose[0], width=0.8)
    plt.xticks(idx, top_ham_terms.keys())
    plt.title("Top words in Normal messages")
    plt.ylabel("Frequency")
    plt.xlabel("Word")
    annotate_bars(ham_bars)
    plt.ylim(0, max(top_ham_terms.values()) + 15)
    plt.savefig(CONFIG["ham_output"])
    plt.close()

    top_spam_terms = {"free": 150, "win": 120, "prize": 90, "account": 60, "earn": 30}
    plt.figure(figsize=(8, 4))
    spam_bars = plt.bar(top_spam_terms.keys(), top_spam_terms.values(), color=rose[1])
    plt.title("Top words in Spam messages")
    plt.ylabel("Frequency")
    plt.xlabel("Word")
    annotate_bars(spam_bars)
    plt.ylim(0, max(top_spam_terms.values()) + 15)
    plt.savefig(CONFIG["spam_output"])
    plt.close()


def render_confusion(y_true, model, vectorizer, text_test):
    cmatrix = confusion_matrix(y_true, model.predict(vectorizer.transform(text_test)))
    plt.figure(figsize=(5, 4))
    pastel_pink_cmap = LinearSegmentedColormap.from_list(
        "pastel_pink",
        CONFIG["pink_cmap"]
    )
    sns.heatmap(
        cmatrix,
        annot=True,
        cmap=pastel_pink_cmap,
        fmt="d",
        vmin=0,
        vmax=cmatrix.max(),
        cbar_kws={"label": "Frequency"}
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(CONFIG["cm_output"])
    plt.close()


def main():
    dataset = load_sms_dataset(CONFIG["data_file"])
    model, vectorizer, x_test, y_test = train_classifier(dataset)

    print(classify_messages([
        "Congratulations! You’ve won a free $1000 gift card! Click the link to claim now.",
        "Hey, just checking if we’re still on for lunch at 1pm today?",
        "URGENT! Your account has been compromised. Visit http://fake-bank.com to secure it.",
        "Are you free tonight lol.",
        "You’ve been selected for a limited time offer—reply YES to receive your discount!"
    ], model, vectorizer))

    render_charts()
    render_confusion(y_test, model, vectorizer, x_test)


if __name__ == "__main__":
    main()
