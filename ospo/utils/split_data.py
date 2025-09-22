import os, random, json
import argparse
from collections import defaultdict

def split_data(data, split_ratio: float = 0.1, ref: str = None): # reference_data
    # data = read_json(data_path)
    if isinstance(data, str):
        data_path = data
        with open(data_path, 'r') as f:
            data = json.load(f)
        total_length = len(data)

    elif isinstance(data, list):
        total_length = len(data)

    if ref is not None:
        with open(ref, 'r') as f:
            ref_data = json.load(f)

        # assert len(ref_data) == len(data) * (1-split_ratio), f"Reference data length {len(ref_data)} does not match expected length {len(data) * (1 - split_ratio)}."
        ref_item_id_list = [item['item_id'] for item in ref_data] 
        
        train_data = []
        val_data = []

        for sample in data:
            item_id = sample['item_id']
            if item_id in ref_item_id_list:
                train_data.append(sample)
            else:
                val_data.append(sample) 

        print(f"Total samples: {total_length}, Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        return val_data, train_data

    # Group samples by t2i_category
    category_dict = defaultdict(list)
    for sample in data:
        category = sample['t2i_category']
        category_dict[category].append(sample)

    val_data = []
    train_data = []

    for category, samples in category_dict.items():
        num_split = int(len(samples) * split_ratio)
        random.shuffle(samples)

        val_data.extend(samples[:num_split])
        train_data.extend(samples[num_split:])


    val_data = sorted(val_data, key=lambda x: x['item_id'])
    train_data = sorted(train_data, key=lambda x: x['item_id'])

    print(f"Total samples: {total_length}, Split samples: {len(val_data)}, Remaining samples: {len(train_data)}")
    
    return val_data, train_data



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets.")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--ref_path', type=str, default=None, help="Reference data is Train Data.")
    parser.add_argument('--directory', type=str, default=None)
    parser.add_argument('--split_ratio', type=float, default=0.1, help="Ratio of validation set size to total dataset size.")
    args = parser.parse_args()

    val_data, train_data = split_data(args.data_path, split_ratio=args.split_ratio, ref=args.ref_path)

    if args.directory is None:
        args.directory = os.path.dirname(args.data_path)
    os.makedirs(args.directory, exist_ok=True)
    
    val_data_path = os.path.join(args.directory, f'val_data_{len(val_data)}.json')
    train_data_path = os.path.join(args.directory, f'train_data_{len(train_data)}.json')

    with open(val_data_path, 'w') as f:
        json.dump(val_data, f, indent=4)
    with open(train_data_path, 'w') as f:
        json.dump(train_data, f, indent=4)
    print("Done.")