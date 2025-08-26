from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

def train_model():
    # ----------------- Paths -----------------
    processed_path = Path(r"C:\Users\bhara\emotion-diary-ai\data\processed\combined_data_clean.csv")
    artifact_dir = Path(r"C:\Users\bhara\emotion-diary-ai\backend\artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "emotion_model.pkl"

    # ----------------- Load Data -----------------
    df = pd.read_csv(processed_path)
    X = df["clean_text"].fillna("").astype(str)
    y = df["status"]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # ----------------- Baseline Model (on the test set for a fair comparison) -----------------
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test) # <-- CHANGE: Predict on the test set
    print("📉 Dummy baseline performance on the test set:")
    print(classification_report(y_test, y_pred_dummy, zero_division=0)) # <-- CHANGE: Evaluate on the test set


    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        # SMOTE will be applied to the training data of each fold in the cross-validation
        ("smote", SMOTE(random_state=42)), 
        ("clf", LogisticRegression(max_iter=2000)) # We can remove class_weight since SMOTE handles the balancing
    ])

    param_grid = {
        "tfidf__max_features": [1000, 2000, 5000],
        "tfidf__ngram_range": [(1,1), (1,2)],
        "clf__C": [0.1, 1, 10]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro", # f1_macro is a great choice for imbalanced problems
        n_jobs=-1,
        verbose=2
    )

    # <-- CHANGE: Fit the grid search ONLY on the training data
    grid.fit(X_train, y_train)

    # <-- CHANGE: Make final predictions on the unseen test data
    y_pred_lr = grid.best_estimator_.predict(X_test)

    # ----------------- Print Best Params and Performance -----------------
    print("⭐ Best model parameters:", grid.best_params_)
    print("⭐ Best CV F1-macro score:", grid.best_score_)
    print("\n📊 Logistic Regression Performance on the TEST SET:")
    # <-- CHANGE: Evaluate final performance on the test set
    print(classification_report(y_test, y_pred_lr))

    # ----------------- Numeric Metrics on Test Set -----------------
    print("\n🔢 Numeric Metrics Comparison on Test Set:")
    print("\nLogistic Regression:")
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print("Precision (macro):", precision_score(y_test, y_pred_lr, average="macro"))
    print("Recall (macro):", recall_score(y_test, y_pred_lr, average="macro"))
    print("F1 Score (macro):", f1_score(y_test, y_pred_lr, average="macro"))

    print("\nDummy Classifier:")
    print("Accuracy:", accuracy_score(y_test, y_pred_dummy))
    print("Precision (macro):", precision_score(y_test, y_pred_dummy, average="macro", zero_division=0))
    print("Recall (macro):", recall_score(y_test, y_pred_dummy, average="macro"))
    print("F1 Score (macro):", f1_score(y_test, y_pred_dummy, average="macro", zero_division=0))

    # ----------------- Confusion Matrix on Test Set -----------------
    cm = confusion_matrix(y_test, y_pred_lr, labels=grid.best_estimator_.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.best_estimator_.classes_)
    
    fig, ax = plt.subplots(figsize=(10, 8)) # Create figure and axes for better control
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix - Logistic Regression (Test Set)")
    plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
    plt.tight_layout() # Adjust layout
    plt.savefig(artifact_dir / "confusion_matrix.png")
    plt.close()
    print(f"📊 Confusion matrix saved at {artifact_dir / 'confusion_matrix.png'}")

    # ----------------- Save Model -----------------
    joblib.dump(grid.best_estimator_, model_path)
    print(f"✅ Model saved at {model_path}")

if __name__ == "__main__":
    train_model()