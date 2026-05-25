import sys
import os

from backend.src.predict import BASE_DIR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "webapp"))


def cmd_preprocess(_):
    from backend.src.data_preprocessing import split_and_copy
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    split_and_copy(
        os.path.join(BASE_DIR, "raw_images"),
        os.path.join(BASE_DIR, "dataset"),
    )


def cmd_train(_):
    from backend.src.train_model import train
    train()


def cmd_predict(args):
    if not args:
        print("Usage: python main.py predict <image_path>")
        return
    from backend.src.predict import predict
    r = predict(args[0])

    if r.get("error") == "not_cardamom":
        print(f"\n⚠️  {r['message']}")
        return

    print(f"\n{'='*52}")
    print(f"  ENSEMBLE PREDICTION RESULT")
    print(f"{'='*52}")
    print(f"  Disease        : {r['disease'].upper().replace('_', ' ')}")
    print(f"  Nepali         : {r['nepali']}")
    print(f"  Confidence     : {r['confidence']}%  ({r['confidence_label']})")
    print(f"  EfficientNet   : {r['model_details']['efficientnet_top']}%")
    print(f"  MobileNet      : {r['model_details']['mobilenet_top']}%")
    print(f"  Severity       : {r['severity']}")
    print(f"\n  {r['description']}")
    print(f"\n  Recommendation: {r['recommendation']}")


def cmd_evaluate(_):
    from backend.src.utils import evaluate_ensemble
    test_dir = os.path.join(BASE_DIR, "dataset", "test")
    evaluate_ensemble(test_dir)


def cmd_web(_):
    from backend.app import create_app
    application = create_app()
    print("\n🌿 Cardamom Disease Detection System")
    print("   Ensemble: EfficientNetB0 + MobileNetV2")
    print("   Web: http://127.0.0.1:5000\n")
    application.run(debug=False, host="0.0.0.0", port=5000)


COMMANDS = {
    "preprocess": cmd_preprocess,
    "train":      cmd_train,
    "predict":    cmd_predict,
    "evaluate":   cmd_evaluate,
    "web":        cmd_web,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0)
    COMMANDS[sys.argv[1]](sys.argv[2:])