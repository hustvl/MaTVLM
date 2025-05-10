import argparse
import json
from collections import defaultdict
import numpy as np

def eval_model(args):
	per_type_acc = defaultdict(list)
	all_acc = []
	
	results = [json.loads(line) for line in open(args.output_path)]
	for sample in results:
		sample['correct'] = sample["answer"].lower().split(".")[0] == sample["text"].split(".")[0].split(",")[0].strip().lower()
                                   
		if "type" in sample:
			per_type_acc[sample["type"]].append(sample['correct'])
			if "object identification" in sample["type"]:
				per_type_acc["object identification"].append(sample['correct'])
			if "spatial" in sample["type"]:
				per_type_acc["spatial relationship understanding"].append(sample['correct'])
		all_acc.append(sample['correct'])
	
	if len(per_type_acc) > 0:
		for test_type in list(per_type_acc.keys()):
			print(test_type, np.mean(per_type_acc[test_type]))
		
	print("All", np.mean(all_acc))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--output-path", type=str, default="eval_result.json")

	args = parser.parse_args()
	eval_model(args)