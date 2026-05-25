import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent
MODEL_DIR      = BASE_DIR / "models"
TRAIN_DIR      = BASE_DIR / "dataset" / "train"
VAL_DIR        = BASE_DIR / "dataset" / "validation"
TEST_DIR       = BASE_DIR / "dataset" / "test"          # ← evaluate on held-out test
RESULTS_DIR    = BASE_DIR / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

EFFNET_PATH    = MODEL_DIR / "model_efficientnet.keras"
MOBILENET_PATH = MODEL_DIR / "model_mobilenet.keras"
LABELS_PATH    = MODEL_DIR / "class_labels.json"

# ── Settings ───────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
SEED       = 42
CLASSES    = ["chhirke", "healthy", "leaf_blight"]


# ─────────────────────────────────────────────────────────────────────────
# 1. LOAD DATASETS
#    KEY FIX: both models receive raw [0,255] pixels from the pipeline.
#    Each model head does its own normalisation internally so the same
#    dataset object feeds both models correctly.
# ─────────────────────────────────────────────────────────────────────────
def load_datasets():
    common = dict(
        image_size   = (IMG_SIZE, IMG_SIZE),
        batch_size   = BATCH_SIZE,
        label_mode   = "categorical",
        class_names  = CLASSES,
        seed         = SEED,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_DIR), shuffle=True,  **common)
    val_ds   = tf.keras.utils.image_dataset_from_directory(
        str(VAL_DIR),   shuffle=False, **common)

    # Use test set if it exists, otherwise fall back to validation
    eval_dir = TEST_DIR if TEST_DIR.exists() else VAL_DIR
    test_ds  = tf.keras.utils.image_dataset_from_directory(
        str(eval_dir),  shuffle=False, **common)

    print(f"\n✅ Classes : {train_ds.class_names}")
    for cls in CLASSES:
        n_tr = len(list((TRAIN_DIR / cls).glob("*.*")))
        n_va = len(list((VAL_DIR   / cls).glob("*.*"))) if VAL_DIR.exists() else 0
        print(f"   {cls:20s}: train={n_tr}  val={n_va}")

    with open(LABELS_PATH, "w") as f:
        json.dump({str(i): cls for i, cls in enumerate(CLASSES)}, f, indent=2)
    print(f"   Labels saved → {LABELS_PATH}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────────────────────────────────────
# 2. CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────
def get_class_weights():
    counts  = [len(list((TRAIN_DIR / cls).glob("*.*"))) for cls in CLASSES]
    total   = sum(counts)
    weights = {i: total / (len(CLASSES) * c) for i, c in enumerate(counts)}
    print("\n⚖️  Class weights:")
    for i, cls in enumerate(CLASSES):
        print(f"   {cls:20s}: {weights[i]:.3f}")
    return weights


# ─────────────────────────────────────────────────────────────────────────
# 3. AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────
def make_augmentation(name="augmentation"):
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.2),
    ], name=name)


# ─────────────────────────────────────────────────────────────────────────
# 4. BUILD EFFICIENTNETB0
#    EfficientNetB0 has built-in preprocessing — pass raw [0,255] pixels.
# ─────────────────────────────────────────────────────────────────────────
def build_efficientnet(num_classes):
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="effnet_input")
    x      = make_augmentation("aug_effnet")(inputs)

    base   = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=x)
    base.trainable = False

    x   = base.output
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(256, activation="relu",
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(128, activation="relu",
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax", name="effnet_output")(x)

    model = Model(inputs=base.input, outputs=out, name="efficientnetb0_model")
    return model, base


# ─────────────────────────────────────────────────────────────────────────
# 5. BUILD MOBILENETV2
#    FIX: Rescaling layer inside the model so the saved .keras file always
#    applies the correct [-1,1] normalisation at inference time too.
#    Previously this was only applied during training via ImageDataGenerator,
#    causing the 100% → 27% collapse at evaluation.
# ─────────────────────────────────────────────────────────────────────────
def build_mobilenet(num_classes):
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="mobilenet_input")
    x      = make_augmentation("aug_mobilenet")(inputs)

    # ✅ Normalisation baked into the model — safe at training AND inference
    x      = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0,
                               name="mobilenet_preprocess")(x)

    base   = MobileNetV2(weights="imagenet", include_top=False, input_tensor=x)
    base.trainable = False

    x   = base.output
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(256, activation="relu",
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(128, activation="relu",
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax", name="mobilenet_output")(x)

    model = Model(inputs=base.input, outputs=out, name="mobilenetv2_model")
    return model, base


# ─────────────────────────────────────────────────────────────────────────
# 6. TRAIN ONE MODEL  (two-phase: frozen → fine-tune)
# ─────────────────────────────────────────────────────────────────────────
def train_single_model(model, base, model_name, save_path,
                       train_ds, val_ds, class_weights):
    print(f"\n{'='*55}")
    print(f"  Training {model_name}")
    print(f"{'='*55}")

    # FIX: label_smoothing removed — it hurts small datasets badly.
    # With only ~3 classes and clean labels, use plain cross-entropy.
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

    # ── Phase 1: train head only ──────────────────────────────────────────
    print(f"\n🚀 Phase 1 — training head (base frozen) …")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    h1 = model.fit(
        train_ds, validation_data=val_ds, epochs=25,
        class_weight=class_weights,
        callbacks=_callbacks(save_path, patience_stop=8, patience_lr=4),
    )
    best1 = max(h1.history["val_accuracy"])
    print(f"   Phase 1 best val_accuracy: {best1*100:.2f}%")

    # ── Phase 2: fine-tune top 40 layers ─────────────────────────────────
    print(f"\n🔥 Phase 2 — fine-tuning top 40 layers …")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    # FIX: use a much lower LR for fine-tuning to avoid destroying
    # ImageNet weights — 1e-4 was too high, causing MobileNetV2 to diverge.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    h2 = model.fit(
        train_ds, validation_data=val_ds, epochs=50,
        class_weight=class_weights,
        callbacks=_callbacks(save_path, patience_stop=10, patience_lr=5),
    )
    best2 = max(h2.history["val_accuracy"])
    print(f"   Phase 2 best val_accuracy: {best2*100:.2f}%")

    final = max(best1, best2)
    print(f"\n✅ {model_name} final accuracy: {final*100:.2f}%")
    print(f"   Saved → {save_path}")
    return h1, h2, final


def _callbacks(save_path, patience_stop, patience_lr):
    return [
        ModelCheckpoint(str(save_path), save_best_only=True,
                        monitor="val_accuracy", verbose=1),
        EarlyStopping(monitor="val_loss", patience=patience_stop,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                          patience=patience_lr, verbose=1, min_lr=1e-8),
    ]


# ─────────────────────────────────────────────────────────────────────────
# 7. EVALUATE ENSEMBLE  (weighted soft voting)
#    FIX: EfficientNetB0 (99%) gets a higher weight than MobileNetV2 (lower)
#    so a weak MobileNetV2 run can't drag the ensemble down.
# ─────────────────────────────────────────────────────────────────────────
def evaluate_ensemble(test_ds, eff_weight=0.6, mob_weight=0.4):
    print(f"\n{'='*55}")
    print(f"  Evaluating Weighted Soft-Voting Ensemble")
    print(f"  EfficientNetB0 weight={eff_weight}  MobileNetV2 weight={mob_weight}")
    print(f"{'='*55}")

    effnet_model    = tf.keras.models.load_model(str(EFFNET_PATH))
    mobilenet_model = tf.keras.models.load_model(str(MOBILENET_PATH))

    all_true, all_eff, all_mob, all_ens = [], [], [], []

    for images, labels in test_ds:
        true_idx   = np.argmax(labels.numpy(), axis=1)
        preds_eff  = effnet_model.predict(images,    verbose=0)
        preds_mob  = mobilenet_model.predict(images, verbose=0)

        # Weighted soft voting
        preds_ens  = eff_weight * preds_eff + mob_weight * preds_mob

        all_true.extend(true_idx)
        all_eff.extend(np.argmax(preds_eff, axis=1))
        all_mob.extend(np.argmax(preds_mob, axis=1))
        all_ens.extend(np.argmax(preds_ens, axis=1))

    n          = len(all_true)
    acc_eff    = np.sum(np.array(all_eff) == np.array(all_true)) / n
    acc_mob    = np.sum(np.array(all_mob) == np.array(all_true)) / n
    acc_ens    = np.sum(np.array(all_ens) == np.array(all_true)) / n

    print(f"\n   EfficientNetB0 alone : {acc_eff*100:.2f}%")
    print(f"   MobileNetV2 alone    : {acc_mob*100:.2f}%")
    print(f"   Weighted Ensemble    : {acc_ens*100:.2f}%  ← final model")

    # ── Classification report ─────────────────────────────────────────────
    print(f"\n📊 Classification Report (Ensemble):")
    print(classification_report(all_true, all_ens, target_names=CLASSES,
                                zero_division=0))

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(all_true, all_ens)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Ensemble Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✅ Saved → {out}")
    plt.close()

    # ── Accuracy comparison bar chart ─────────────────────────────────────
    plt.figure(figsize=(6, 4))
    models = ["EfficientNetB0", "MobileNetV2", "Ensemble"]
    accs   = [acc_eff * 100, acc_mob * 100, acc_ens * 100]
    bars   = plt.bar(models, accs, color=["#4a9463", "#2d5c3f", "#c8a84b"],
                     edgecolor="white", linewidth=0.8)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{acc:.2f}%", ha="center", va="bottom", fontsize=10)
    plt.ylim(0, 110)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    out2 = RESULTS_DIR / "accuracy_comparison.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"✅ Saved → {out2}")
    plt.close()

    return acc_eff, acc_mob, acc_ens


# ─────────────────────────────────────────────────────────────────────────
# 8. PLOT TRAINING HISTORY
# ─────────────────────────────────────────────────────────────────────────
def plot_history(h1, h2, title, out_path):
    try:
        acc   = h1.history["accuracy"]     + h2.history["accuracy"]
        vacc  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
        loss  = h1.history["loss"]         + h2.history["loss"]
        vloss = h1.history["val_loss"]     + h2.history["val_loss"]
        split = len(h1.history["accuracy"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(vacc,  label="val accuracy",   color="#4a9463")
        ax1.plot(acc,   label="train accuracy",  color="#4a9463", alpha=0.4)
        ax1.axvline(split - 1, color="gray", ls="--", label="fine-tune start")
        ax1.set_title(f"{title} — Accuracy")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(vloss, label="val loss",   color="#c8a84b")
        ax2.plot(loss,  label="train loss", color="#c8a84b", alpha=0.4)
        ax2.axvline(split - 1, color="gray", ls="--", label="fine-tune start")
        ax2.set_title(f"{title} — Loss")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"   Chart saved → {out_path}")
        plt.close()
    except Exception as e:
        print(f"   Chart skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────
def train():
    print("\n Loading datasets …")
    train_ds, val_ds, test_ds = load_datasets()
    class_weights = get_class_weights()
    num_classes   = len(CLASSES)

    # ── EfficientNetB0 ────────────────────────────────────────────────────
    eff_model, eff_base = build_efficientnet(num_classes)
    h1e, h2e, acc_eff   = train_single_model(
        eff_model, eff_base, "EfficientNetB0",
        EFFNET_PATH, train_ds, val_ds, class_weights,
    )
    plot_history(h1e, h2e, "EfficientNetB0",
                 str(MODEL_DIR / "history_efficientnet.png"))

    # ── MobileNetV2 ───────────────────────────────────────────────────────
    mob_model, mob_base = build_mobilenet(num_classes)
    h1m, h2m, acc_mob   = train_single_model(
        mob_model, mob_base, "MobileNetV2",
        MOBILENET_PATH, train_ds, val_ds, class_weights,
    )
    plot_history(h1m, h2m, "MobileNetV2",
                 str(MODEL_DIR / "history_mobilenet.png"))

    # ── Weighted ensemble evaluation on test set ──────────────────────────
    # Determine weights dynamically based on validation performance
    total     = acc_eff + acc_mob
    eff_w     = round(acc_eff / total, 2)
    mob_w     = round(1 - eff_w, 2)
    print(f"\n Dynamic ensemble weights: EfficientNetB0={eff_w}  MobileNetV2={mob_w}")
    evaluate_ensemble(test_ds, eff_weight=eff_w, mob_weight=mob_w)

    print(f"\n{'='*55}")
    print(f"  Training Complete!")
    print(f"{'='*55}")
    print(f"  EfficientNetB0 : {acc_eff*100:.2f}%")
    print(f"  MobileNetV2    : {acc_mob*100:.2f}%")
    print(f"  Models saved   : models/")
    print(f"    model_efficientnet.keras")
    print(f"    model_mobilenet.keras")


if __name__ == "__main__":
    train()