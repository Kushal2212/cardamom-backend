import os
import numpy as np
import tensorflow as tf
import cv2

IMG_SIZE = 224


# ─────────────────────────────────────────────────────────────
# 🔍 Find last conv layer automatically
# ─────────────────────────────────────────────────────────────
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")


# ─────────────────────────────────────────────────────────────
# 🔥 Grad-CAM++ implementation
# ─────────────────────────────────────────────────────────────
def generate_gradcam_pp(img_path: str, model, class_idx: int, output_path: str):
    try:
        # ── Load image ─────────────────────────────────────────
        img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        arr = tf.keras.utils.img_to_array(img)
        arr = np.expand_dims(arr, axis=0).astype("float32")

        # ── Get last conv layer ────────────────────────────────
        last_conv_name = find_last_conv_layer(model)

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )

        # ── Gradient computation ──────────────────────────────
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(arr)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            raise ValueError("Gradients are None")

        conv_outputs = conv_outputs[0]
        grads = grads[0]

        # ── Grad-CAM++ weights ───────────────────────────────
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3

        sum_activations = tf.reduce_sum(conv_outputs, axis=(0, 1))

        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_activations * grads_power_3
        alpha_denom = tf.where(alpha_denom != 0, alpha_denom, tf.ones_like(alpha_denom))

        alphas = alpha_num / alpha_denom

        positive_gradients = tf.nn.relu(grads)
        weights = tf.reduce_sum(alphas * positive_gradients, axis=(0, 1))

        # ── Compute heatmap ───────────────────────────────────
        heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)

        heatmap = tf.nn.relu(heatmap)
        heatmap /= tf.reduce_max(heatmap) + 1e-8
        heatmap = heatmap.numpy()

        # ── Resize heatmap ───────────────────────────────────
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_uint8 = np.uint8(255 * heatmap)

        # ── Apply colormap ───────────────────────────────────
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # ── Overlay ──────────────────────────────────────────
        original = cv2.imread(img_path)
        original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

        overlay = cv2.addWeighted(original, 0.75, heatmap_color, 0.25, 0)

        # ── Add legend ───────────────────────────────────────
        bar_h = 18
        canvas = np.zeros((IMG_SIZE + bar_h + 22, IMG_SIZE, 3), dtype=np.uint8)

        canvas[:IMG_SIZE] = overlay

        for x in range(IMG_SIZE):
            val = int(x / IMG_SIZE * 255)
            color = cv2.applyColorMap(
                np.array([[val]], dtype=np.uint8),
                cv2.COLORMAP_JET
            )[0][0]
            canvas[IMG_SIZE + 4: IMG_SIZE + 4 + bar_h, x] = color

        cv2.putText(canvas, "Low",
                    (2, IMG_SIZE + bar_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (180, 180, 180), 1)

        cv2.putText(canvas, "High",
                    (IMG_SIZE - 30, IMG_SIZE + bar_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (180, 180, 180), 1)

        # ── Save ─────────────────────────────────────────────
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, canvas)

        return output_path

    except Exception as e:
        print(f"Grad-CAM++ error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# 🔧 Auto version (recommended)
# ─────────────────────────────────────────────────────────────
def generate_gradcam_pp_auto(img_path, model, preds, output_path):
    class_idx = int(np.argmax(preds))
    return generate_gradcam_pp(img_path, model, class_idx, output_path)