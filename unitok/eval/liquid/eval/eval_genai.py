import t2v_metrics
import os
from tqdm import tqdm
import torch

import numpy as np
import json

import argparse


def main(prompt_file, tag_file, image_dir):
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl')

    with open(prompt_file, 'r') as f:
        lines = f.readlines()
    all_prompts = []
    for index, line in enumerate(lines):
        all_prompts.append({'Index': str(index + 1).zfill(5), 'Prompt': line.strip()})

    pair_list = []
    all_pair_list = []
    for i in range(len(all_prompts)):
        data = all_prompts[i]
        pair_list.append(
            {'images': [image_dir + '{}.jpg'.format(str(data['Index']))], 'texts': [data['Prompt']]}
        )
        if len(pair_list) == 16:
            all_pair_list.append(pair_list)
            pair_list = []
    if len(pair_list) > 0:
        all_pair_list.append(pair_list)

    print('loading:', image_dir)
    score_list = []
    for pair_list in all_pair_list:
        scores = clip_flant5_score.batch_forward(dataset=pair_list, batch_size=len(pair_list))  # (n_sample, 4, 1) tensor
        score_list.append(scores.squeeze())
    all_score = torch.cat(score_list)

    our_scores = all_score
    tag_groups = {
        'basic': ['attribute', 'scene', 'spatial relation', 'action relation', 'part relation', 'basic'],
        'advanced': ['counting', 'comparison', 'differentiation', 'negation', 'universal', 'advanced'],
        'overall': ['basic', 'advanced', 'all']
    }


    tag_result = {}
    tags = json.load(open(tag_file))
    prompt_to_items = {str(i).zfill(5): [i - 1] for i in range(1, 528)}
    items_by_model_tag = {}
    for tag in tags:
        items_by_model_tag[tag] = {}
        for prompt_idx in tags[tag]:
            for image_idx in prompt_to_items[f"{prompt_idx:05d}"]:
                model = 'my model'
                if model not in items_by_model_tag[tag]:
                    items_by_model_tag[tag][model] = []
                items_by_model_tag[tag][model].append(image_idx)

    for tag in tags:
        tag_result[tag] = {}
        for model in items_by_model_tag[tag]:
            our_scores_mean = our_scores[items_by_model_tag[tag][model]].mean()
            our_scores_std = our_scores[items_by_model_tag[tag][model]].std()
            tag_result[tag][model] = {
                'metric': {'mean': our_scores_mean, 'std': our_scores_std},
            }

    tag_result['all'] = {}
    all_models = items_by_model_tag[tag]
    for model in all_models:
        all_model_indices = set()
        for tag in items_by_model_tag:
            all_model_indices = all_model_indices.union(set(items_by_model_tag[tag][model]))
        all_model_indices = list(all_model_indices)
        our_scores_mean = our_scores[all_model_indices].mean()
        our_scores_std = our_scores[all_model_indices].std()
        tag_result['all'][model] = {
            'metric': {'mean': our_scores_mean, 'std': our_scores_std},
        }

    for tag_group in tag_groups:
        for score_name in ['metric']:
            print(f"Tag Group: {tag_group} ({score_name} performance)")
            tag_header = f"{'Model':<20}" + " ".join([f"{tag:<20}" for tag in tag_groups[tag_group]])
            print(tag_header)
            for model_name in all_models:
                detailed_scores = [
                    f"{tag_result[tag][model_name][score_name]['mean']:.2f} +- {tag_result[tag][model_name][score_name]['std']:.2f}"
                    for tag in tag_groups[tag_group]]
                detailed_scores = " ".join([f"{score:<20}" for score in detailed_scores])
                model_scores = f"{model_name:<20}" + detailed_scores
                print(model_scores)
            print()
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file', type=str)
    parser.add_argument('--tag_file', type=str)
    parser.add_argument('--image_dir', type=str)
    args = parser.parse_args()
    main(args.prompt_file, args.tag_file, args.image_dir)

"""
pip install t2v-metrics    
pip install git+https://github.com/openai/CLIP.git
sudo apt-get install libgl1
"""