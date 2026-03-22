import argparse
import os

from trainer import ToothbrushDefectDetector
from evaluator import ToothbrushModelEvaluator

def main():
    parser = argparse.ArgumentParser(description="U-Net Model for Toothbrush Defect Detection")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "eval"], 
        required=True, 
        help="Execution mode: 'train' to train a new model, 'eval' to evaluate an existing one"
    )
    parser.add_argument(
        "-v", "--version", 
        type=str, 
        default="1", 
        help="Model version string for saving/loading checkpoints (default: '1')"
    )
    parser.add_argument(
        "-e", "--epochs", 
        type=int, 
        default=20, 
        help="Number of training epochs (default: 20)"
    )
    
    args = parser.parse_args()

    if args.mode == "train":
        print(f"Starting training process for {args.epochs} epochs (Version: {args.version})...")
        detector = ToothbrushDefectDetector()
        detector.prepare_data()
        detector.train_unet(epochs=args.epochs, version=args.version)
        print("Training finished.")

    elif args.mode == "eval":
        model_path = f"checkpoint_toothbrush_unet_v{args.version}.pth"
        
        if not os.path.exists(model_path):
            print(f"Error: Model checkpoint '{model_path}' not found.")
            print("Please train the model first or check the version number.")
            return

        print(f"Evaluating model: {model_path}...")
        evaluator = ToothbrushModelEvaluator(model_path)
        evaluator.evaluate_dataset()
        evaluator.visualize_examples()

if __name__ == "__main__":
    main()