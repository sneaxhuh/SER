import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from ser_utils import CRNN_SER, extract_mfcc, pad_or_truncate, predict, NUM_CLASSES

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single audio file")
    group.add_argument("--folder", type=str, help="Path to a folder containing audio files")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--output", type=str, help="CSV output filename (used with --folder)")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CRNN_SER(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")

    # -------------------------------
    # Single File Mode
    # -------------------------------
    if args.file:
        try:
            mfcc = extract_mfcc(args.file)
            tensor = pad_or_truncate(mfcc)
            emotion = predict(model, tensor, device)
            print(emotion)
        except Exception:
            print("Error")

    # -------------------------------
    # Folder Mode
    # -------------------------------
    elif args.folder:
        results = []
        audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
        audio_files = [f for f in os.listdir(args.folder) if f.lower().endswith(audio_extensions)]

        print(f"🔍 Found {len(audio_files)} audio files. Starting prediction...\n")

        for fname in tqdm(audio_files, desc="Processing"):
            fpath = os.path.join(args.folder, fname)
            try:
                mfcc = extract_mfcc(fpath)
                tensor = pad_or_truncate(mfcc)
                emotion = predict(model, tensor, device)
                results.append((fname, emotion))
            except Exception:
                results.append((fname, "error"))

        if args.output:
            df = pd.DataFrame(results, columns=["file", "predicted_emotion"])
            df.to_csv(args.output, index=False)
            print(f"\n✅ All files processed. Results saved to: {args.output}")

if __name__ == "__main__":
    main()
