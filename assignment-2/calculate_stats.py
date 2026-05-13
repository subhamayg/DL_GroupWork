import json
import math
from collections import defaultdict

def calculate_stats(file_path, num_folds=5):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            model = entry['model']
            diff = entry['difficulty']
            length = entry['context_length']
            pos = entry['evidence_position']
            # Treat errors or null as incorrect
            is_correct = entry.get('is_correct', False)
            if is_correct is None: is_correct = False
            data[model][diff][length][pos].append(int(is_correct))

    results = {}
    for model in data:
        results[model] = {}
        for diff in data[model]:
            results[model][diff] = {}
            for length in data[model][diff]:
                results[model][diff][length] = {}
                for pos in data[model][diff][length]:
                    samples = data[model][diff][length][pos]
                    # Split into folds
                    fold_size = len(samples) // num_folds
                    fold_accuracies = []
                    for i in range(num_folds):
                        fold = samples[i*fold_size : (i+1)*fold_size]
                        acc = sum(fold) / len(fold)
                        fold_accuracies.append(acc)
                    
                    mean = sum(fold_accuracies) / len(fold_accuracies)
                    variance = sum((x - mean) ** 2 for x in fold_accuracies) / len(fold_accuracies)
                    std = math.sqrt(variance)
                    
                    results[model][diff][length][pos] = {
                        "mean": round(mean, 2),
                        "std": round(std, 2)
                    }
    return results

if __name__ == "__main__":
    stats = calculate_stats('target/Assignment/positional_bias_results.jsonl')
    with open('analysis/data/stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("Stats calculated and saved to analysis/data/stats.json")
