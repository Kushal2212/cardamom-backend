import os
import json
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")

CLASSES = ["chhirke", "healthy", "leaf_blight"]


# ─────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────
def get_class_labels() -> dict:
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError("class_labels.json not found. Train first.")
    with open(LABELS_PATH) as f:
        return json.load(f)


def get_index_to_class() -> dict:
    labels = get_class_labels()
    first_key = next(iter(labels))
    if str(first_key).isdigit():
        return {int(k): v for k, v in labels.items()}
    return {int(v): k for k, v in labels.items()}



def evaluate_ensemble(test_dir):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score)

    effnet_path    = os.path.join(MODEL_DIR, "model_efficientnet.keras")
    mobilenet_path = os.path.join(MODEL_DIR, "model_mobilenet.keras")

    for path in (effnet_path, mobilenet_path):
        if not os.path.exists(path):
            print(f"❌ Missing: {path}\nTrain first: python main.py train")
            return

    print("Loading EfficientNetB0...")
    effnet    = load_model(effnet_path)
    print("Loading MobileNetV2...")
    mobilenet = load_model(mobilenet_path)

    # ── Use test dir if given, else fall back to validation ───────────────
    if not os.path.isdir(test_dir):
        val_dir = os.path.join(BASE_DIR, "dataset", "validation")
        print(f"⚠️  test_dir not found, falling back to: {val_dir}")
        test_dir = val_dir

    # ── Load dataset — NO external rescaling ─────────────────────────────
    # image_dataset_from_directory returns float32 [0, 255] by default.
    # class_names is fixed so order is always chhirke/healthy/leaf_blight.
    IMG_SIZE   = 224
    BATCH_SIZE = 32

    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size  = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        label_mode  = "categorical",
        class_names = CLASSES,      # ← fixed order, no shuffle surprise
        shuffle     = False,
        seed        = 42,
    )
    class_names = ds.class_names
    total       = sum(1 for _ in ds.unbatch())
    print(f"\nClasses found: {class_names}")
    print(f"Total test images: {total}\n")

    ds = ds.cache().prefetch(tf.data.AUTOTUNE)

    # ── Collect predictions ───────────────────────────────────────────────
    y_true, y_pred_eff, y_pred_mob, y_pred_ens = [], [], [], []
    steps = -(-total // BATCH_SIZE)   # ceiling division

    for step, (images, labels) in enumerate(ds, 1):
        true_idx  = np.argmax(labels.numpy(), axis=1)

        # ✅ Both models get raw [0,255] — internal Rescaling handles the rest
        p_eff = effnet.predict(images,    verbose=0)
        p_mob = mobilenet.predict(images, verbose=0)
        p_ens = 0.5 * p_eff + 0.5 * p_mob

        y_true.extend(true_idx)
        y_pred_eff.extend(np.argmax(p_eff, axis=1))
        y_pred_mob.extend(np.argmax(p_mob, axis=1))
        y_pred_ens.extend(np.argmax(p_ens, axis=1))
        print(f"  Step {step}/{steps} done", end="\r")

    print()

    # ── Accuracy ──────────────────────────────────────────────────────────
    acc_eff = accuracy_score(y_true, y_pred_eff) * 100
    acc_mob = accuracy_score(y_true, y_pred_mob) * 100
    acc_ens = accuracy_score(y_true, y_pred_ens) * 100

    print(f"\n{'='*50}")
    print(f"  EfficientNetB0  Accuracy : {acc_eff:.2f}%")
    print(f"  MobileNetV2     Accuracy : {acc_mob:.2f}%")
    print(f"  Ensemble        Accuracy : {acc_ens:.2f}%  ← Best")
    print(f"{'='*50}")

    print("\nDetailed Classification Report (Ensemble):")
    print(classification_report(
        y_true, y_pred_ens,
        target_names = class_names,
        zero_division = 0,
    ))

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 3-panel confusion matrix ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, y_pred, title in zip(
        axes,
        [y_pred_eff, y_pred_mob, y_pred_ens],
        ["EfficientNetB0", "MobileNetV2", "Ensemble (Soft Voting)"],
    ):
        cm  = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred) * 100
        im  = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks      = np.arange(len(class_names)),
            yticks      = np.arange(len(class_names)),
            xticklabels = class_names,
            yticklabels = class_names,
            xlabel      = f"Predicted  |  Accuracy: {acc:.2f}%",
            ylabel      = "True label",
            title       = title,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center", fontsize=13, fontweight="bold",
                        color="white" if cm[i, j] > thresh else "black")

    plt.suptitle("Confusion Matrix — Cardamom Disease Detection System",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved → {cm_path}")

    # ── Accuracy bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["EfficientNetB0", "MobileNetV2", "Ensemble"],
        [acc_eff, acc_mob, acc_ens],
        color=["#4a9463", "#2d5c3f", "#c8a84b"],
        width=0.5, edgecolor="white",
    )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
    for bar, acc in zip(bars, [acc_eff, acc_mob, acc_ens]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{acc:.2f}%", ha="center", fontsize=12, fontweight="bold")
    ax.axhline(y=max([acc_eff, acc_mob, acc_ens]), color="red",
               linestyle="--", alpha=0.4,
               label=f"Best: {max([acc_eff, acc_mob, acc_ens]):.2f}%")
    ax.legend()
    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved → {bar_path}")
    print("\nInsert both images from results/ folder into Chapter 5 of your report.")


# ─────────────────────────────────────────────────────────────────────────
# evaluate_model  (single model, kept for compatibility)
# ─────────────────────────────────────────────────────────────────────────
def evaluate_model(model_path, test_dir):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from sklearn.metrics import classification_report, confusion_matrix

    model = load_model(model_path)

    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size  = (224, 224),
        batch_size  = 32,
        label_mode  = "categorical",
        class_names = CLASSES,
        shuffle     = False,
    )
    all_true, all_pred = [], []
    for images, labels in ds:
        preds = model.predict(images, verbose=0)
        all_true.extend(np.argmax(labels.numpy(), axis=1))
        all_pred.extend(np.argmax(preds, axis=1))

    print("\nClassification Report:")
    print(classification_report(all_true, all_pred,
                                target_names=CLASSES, zero_division=0))

    cm      = confusion_matrix(all_true, all_pred)
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    _plot_cm(cm, CLASSES, cm_path)


def _plot_cm(cm, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True label", xlabel="Predicted label",
           title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        evaluate_model(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python src/utils.py <model.keras> <test_dir>")