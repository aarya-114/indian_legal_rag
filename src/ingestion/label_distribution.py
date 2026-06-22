import json
from collections import Counter

with open("data/raw/rhetorical_spans.json") as f:
    spans = json.load(f)

with open("data/raw/judgment_metadata.json") as f:
    metadata = json.load(f)


with open("data/raw/label_distribution.json","w")as f:

    # label distribution
    f.write("=== LABEL DISTRIBUTION ===\n")
    labels=[s['label'] for s in spans]
    for label,count in Counter(labels).most_common():
        f.write(f"{label}:{count}\n")

    #case type distribution
    f.write("\n=== CASE TYPE DISTRIBUTION ===\n")
    case_types=[m['case_type']for m in metadata]
    for ct,count in Counter(case_types).most_common():
        f.write(f"{ct}:{count}\n")
    
    # court type distribution
    f.write("\n=== COURT TYPE DISTRIBUTION ===\n")
    court_types=[m['court_type'] for m in metadata] 
    for ct,count in Counter(court_types).most_common():
        f.write(f"{ct}:{count}\n")
    
    # text length stats
    lengths=[m['text_length'] for m in metadata]
    f.write("\n=== TEXT LENGTH STATS ===\n")
    f.write(f"Min: {min(lengths):,} chars\n")
    f.write(f"Max: {max(lengths):,} chars\n")
    f.write(f"Avg: {sum(lengths)//len(lengths):,} chars\n")


