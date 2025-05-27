import argparse
import json
from pathlib import Path
from src.inference.pipeline import DetectionPipeline
from src.config import config

def main():
    parser = argparse.ArgumentParser(description='Deepfake Video Detection')
    parser.add_argument('input_path', type=str, help='Path to input video file or directory')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_RESULTS_DIR,
                        help='Output directory for results')
    args = parser.parse_args()

    # Resolve the input path to an absolute path
    input_path = Path(args.input_path).resolve()
    print(f"Resolved input path: {input_path}")
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}")
        return

    # Initialize detection pipeline
    print("Initializing detection pipeline...")
    detector = DetectionPipeline()

    results = []
    if input_path.is_file():
        print(f"Analyzing file: {input_path}")
        result = detector.analyze(str(input_path))
        results.append(result)
    elif input_path.is_dir():
        print(f"Analyzing directory: {input_path}")
        for video_file in input_path.glob('*.mp4'):
            print(f"Analyzing file: {video_file}")
            result = detector.analyze(str(video_file))
            results.append(result)
    else:
        print("Input path is neither a file nor a directory.")
        return

    # Print results to console
    for r in results:
        print(r)

    # Save results to the output directory (ensure it exists)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'detection_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Analysis complete. Results saved to {output_path}")

if __name__ == '__main__':
    main()
