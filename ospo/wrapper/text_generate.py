import os
import re
import random
import traceback
import torch
from pytorch_lightning import LightningModule

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.constant import *
from ospo.utils.common import save_json, save_json_ddp, set_seed
from ospo.utils.processor import get_sft_format, get_processor_output, batchify
from ospo.templates import get_prompt_element, get_conversation_element, get_prompt_negative, get_prompt_dense


class JanusProElementGenWrapper(LightningModule):
    def __init__(self, config, model, tokenizer, processor):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor  
        self.category = config.category
        self.max_len = config.max_len
        self.interval = config.interval

        if self.max_len is None:
            if config.category == "2D_spatial" or config.category == "3D_spatial" or config.category == "numeracy1" or config.category == "numeracy2":
                self.max_len = 1000
            elif config.category == "non-spatial" or config.category == "complex":
                self.max_len = 4000
            else:
                self.max_len = 667

        print("# category: ", self.category)
        print('# max_len: ', self.max_len)

        # t2i_category mapping
        self.category_dict = {'color1': 'attribute', 
                                'color2': 'attribute', 
                                'shape1': 'attribute', 
                                'shape2': 'attribute', 
                                'texture1': 'attribute', 
                                'texture2': 'attribute', 
                                '2D_spatial': 'layout', 
                                '3D_spatial': 'layout', 
                                'numeracy1': 'layout', 
                                'numeracy2': 'layout', 
                                'non-spatial': 'non-spatial', 
                                'complex': 'complex'}

        # sub_category mapping
        self.sub_category_dict = {'color1': 'attribute1_color', 
                                'color2': 'attribute2', 
                                'shape1': 'attribute1_shape', 
                                'shape2': 'attribute2', 
                                'texture1': 'attribute1_texture', 
                                'texture2': 'attribute2', 
                                '2D_spatial': 'layout1', 
                                '3D_spatial': 'layout1', 
                                'numeracy1': 'layout2', 
                                'numeracy2': 'layout3', 
                                'non-spatial': 'non-spatial', 
                                'complex': 'complex'}

        self.object_prompt = get_prompt_element(self.category, self.processor)
        self.element_set = set()
        self.element_list = list()


    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if len(self.element_set) >= self.max_len:
            return 

        if batch_idx % self.interval == 0 and batch_idx !=0 :
            self.object_prompt = self.update_fewshot(self.element_list, self.category)
            answers = self.generate(prompt=self.object_prompt)
        else:
            answers = self.generate(prompt=self.object_prompt)

        for answer in answers:
            if self.category not in ["non-spatial", "complex"]:
                stop_words = ['\n', '/', '-', 'color', 'shape', 'texture', 'spatial', 'numeracy', '(', ')']
                try:
                    answer = answer.split(';')
                    prompt, concepts_and_relations, relation = [a.strip().split(':')[-1].strip() for a in answer]
                    
                    if all(stop not in prompt for stop in stop_words):
                        filtered_answer = prompt
                    else:
                        raise Exception("Filtered answer is empty.")

                    if filtered_answer == "":
                        raise Exception("Filtered answer is empty.")
                    
                    if filtered_answer not in self.element_set:
                        self.element_set.add(filtered_answer)
                        self.element_list.append({
                            "t2i_category": self.category_dict[self.category],
                            "sub_category": self.sub_category_dict[self.category],
                            "prompt": filtered_answer.strip(),
                            "concepts_and_relations": concepts_and_relations.strip(),
                            "relation": relation.strip()
                        })
                        
                        print("# Current element_set size:", len(self.element_set))

                except Exception as e:
                    print(f"Failed to process answer: {answer}\nError: {e}")

            else:
                try:
                    answer = answer.split(';')
                    prompt, concepts_and_relations, relation = [a.strip().split(':')[-1].strip() for a in answer]
                    
                    if prompt not in self.element_set:
                        self.element_set.add(prompt)
                        self.element_list.append({
                            "t2i_category": self.category_dict[self.category],
                            "sub_category": self.sub_category_dict[self.category],
                            "prompt": prompt.strip(),
                            "concepts_and_relations": concepts_and_relations.strip(),
                            "relation": relation.strip()
                        })
                except Exception as e:
                    print(f"Failed to process answer: {answer}\nError: {e}")

            if len(self.element_set) >= self.max_len:
                break
            
    @torch.inference_mode()
    def generate(self, prompt):
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).to(self.device)
        input_ids = input_ids.repeat(self.config.repeat, 1)
        
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        attention_mask = torch.ones_like(input_ids)
        
        outputs = self.model.language_model.generate(inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                **self.config.generation_config
                )
        answer = self.tokenizer.batch_decode(outputs.cpu().tolist(), skip_special_tokens=True)

        return answer

    def update_fewshot(self, element_list, category):
        
        conv_updated = []

        system, conversation = get_conversation_element(category, self.processor)
        input_prompt = conversation[0]['content']
        num_few_shot = 5 if category == "complex" else 3

        selected = random.sample(element_list, num_few_shot)
        for s in selected:
            conv_updated.append({
                "role": "<|User|>",
                "content": input_prompt
            })
            conv_updated.append({
                "role": "<|Assistant|>",
                "content": f"Prompt: {s['prompt']}; Concepts and relations: {s['concepts_and_relations']}; Relation: {s['relation']}"
            })
        
        conv_updated.append({
            "role": "<|User|>",
            "content": input_prompt
        })
        conv_updated.append({
            "role": "<|Assistant|>",
            "content": ""
        })

        sft_format_updated = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conv_updated,
            sft_format=self.processor.sft_format,
            system_prompt=system,
        )
        return sft_format_updated
    
    def on_test_epoch_end(self):
        froot = self.config.save_path
        fname = f'{self.category}_prompt'
        save_json(froot, fname, self.element_list)
        print(f'# Generated [{self.category}] elements: {len(self.element_set)}')
 
 

class JanusProNegativeGenWrapper(LightningModule):
    def __init__(self, config, model, tokenizer, processor):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.image_start_id = self.tokenizer.vocab.get("<begin_of_image>")
        self.image_end_id = self.tokenizer.vocab.get("<end_of_image>")
        self.image_id = self.tokenizer.vocab.get("<image_placeholder>")
        self.pad_id = self.tokenizer.vocab.get("<｜▁pad▁｜>")

        self.output_list = []
        

    def on_test_epoch_start(self):
        self.model.eval() 
    

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        try: 
            all_outputs_by_index = [[] for _ in range(3)]  # 3 perturbation type
            
            # 1. group by perturbation index
            grouped_pair_lists = [[] for _ in range(3)]
            for sample in batch:
                prompt = sample['prompt']
                sub_category = sample['sub_category']
                perturbed_methods = sample['perturbed_method']

                for i, p_type in enumerate(perturbed_methods):
                    grouped_pair_lists[i].append((sub_category, prompt, p_type))

            # 2. generate per group with fixed seed
            for i, pair_list in enumerate(grouped_pair_lists):
                set_seed(self.config.seed_list[i])

                input_embeds, batched_prepares = self.prepare_input_embeds(pair_list)
                outputs = self.generate(input_embeds, batched_prepares)
                all_outputs_by_index[i] = outputs

            # 3. regroup to sample-wise format
            for sample_idx in range(len(batch)): # = batch_size
                perturbed_output = []

                for i in range(3):
                    output = all_outputs_by_index[i][sample_idx]
                    answer = self.tokenizer.decode(output.cpu().tolist(), skip_special_tokens=True)
                    answer_output = answer.split("Contrastive Prompt: ")[-1].strip()

                    perturbed_output.append(answer_output)

                batch[sample_idx]['negative_prompt'] = perturbed_output
                self.output_list.append(batch[sample_idx])
            
        except Exception as e:
            print(f"Error in test_step: {e}")
            traceback.print_exc()

            
    def prepare_input_embeds(self, pair_list): 
        # pair_list: list of Tuple (sub_category, prompt, p_type)
        prepare_list = []
        for triplet in pair_list:     
            sub_category, prompt, p_type = triplet
            get_func = get_prompt_negative[sub_category]

            system_prompt, conversation = get_func(p_type, prompt)
            if system_prompt is None or conversation is None:
                print("None system_prompt or conversation")
                continue

            sft_format = get_sft_format(self.processor, system_prompt, conversation)
            prepare_list.append(get_processor_output(self.processor, self.tokenizer, sft_format))
    
        # batchify
        batched_prepares = batchify(self.processor, self.tokenizer, prepare_list)
        batched_prepares = batched_prepares.to(self.device) 
        inputs_embeds = self.model.prepare_inputs_embeds(**batched_prepares)

        return inputs_embeds, batched_prepares


    @torch.inference_mode()
    def generate(self, input_embeds, batched_prepares):
        outputs = self.model.language_model.generate(inputs_embeds=input_embeds,
            attention_mask=batched_prepares.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            **self.config.generation_config
            )

        return outputs

    def on_test_epoch_end(self):
        save_json_ddp(
            save_root=self.config.save_path,
            save_name='negative_prompt' if self.config.save_name is None else self.config.save_name,
            world_size=self.trainer.world_size,
            save_file=self.output_list,
            rank=self.trainer.global_rank,
        )
        # print("Negative prompt generation done.")
        print(f"Negative prompt saved at {os.path.join(self.config.save_path, 'negative_prompt.json')}")



class JanusProDenseGenWrapper(LightningModule):
    def __init__(self, config, model, tokenizer, processor):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.image_start_id = self.tokenizer.vocab.get("<begin_of_image>")
        self.image_end_id = self.tokenizer.vocab.get("<end_of_image>")
        self.image_id = self.tokenizer.vocab.get("<image_placeholder>")
        self.pad_id = self.tokenizer.vocab.get("<｜▁pad▁｜>")

        self.output_list = []


    def on_test_epoch_start(self):
        self.model.eval() 
    

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        try:        
            all_outputs_by_index = [[] for _ in range(3)]  # 3 perturbation type
            
            # 1. group by perturbation index
            skip_flags = [[] for _ in range(3)]  
            grouped_pair_lists = [[] for _ in range(3)]
            
            for sample in batch:
                prompt = sample['prompt']
                sub_category = sample['sub_category']
                negative_prompt_list = sample['negative_prompt'] 

                for i, negative_prompt in enumerate(negative_prompt_list):
                    # Triplet: category, positive(base), negative
                    if negative_prompt == "":
                        skip_flags[i].append(True)
                        # grouped_pair_lists[i].append(("", "", ""))  # Dummy triplet
                        grouped_pair_lists[i].append(None)
                    else:
                        skip_flags[i].append(False)
                        grouped_pair_lists[i].append((sub_category, prompt, negative_prompt))


            # 2. generate per group with fixed seed
            for i, pair_list in enumerate(grouped_pair_lists):
                set_seed(self.config.seed_list[i])

                # input_embeds, batched_prepares = self.prepare_input_embeds(pair_list)
                # outputs = self.generate(input_embeds, batched_prepares)
                # all_outputs_by_index[i] = outputs

                input_embeds, (batched_prepares, valid_indices) = self.prepare_input_embeds(pair_list)

                if input_embeds == []:
                    outputs = [None] * len(pair_list)
                else:
                    partial_outputs = self.generate(input_embeds, batched_prepares)
                    outputs = [None] * len(pair_list)
                    for j, idx in enumerate(valid_indices):
                        outputs[idx] = partial_outputs[j]

                all_outputs_by_index[i] = outputs


            # 3. regroup to sample-wise format
            def post_process(raw_output):
                base_long = re.search(r"Step 2\. Prompt 1 Dense: (.+)", raw_output)
                negative_long = re.search(r"Step 4\. Prompt 2 Dense: (.+)", raw_output)

                base_long_prompt = base_long.group(1) if base_long else ""
                negative_long_prompt = negative_long.group(1) if negative_long else ""
                
                return base_long_prompt, negative_long_prompt

            for sample_idx in range(len(batch)): # = batch_size
                base_long_output = []
                negative_long_output = []

                for i in range(3):
                    if skip_flags[i][sample_idx]: 
                        base_long_prompt = ""
                        negative_long_prompt = ""
                    else:
                        output = all_outputs_by_index[i][sample_idx]
                        answer = self.tokenizer.decode(output.cpu().tolist(), skip_special_tokens=True)
                        base_long_prompt, negative_long_prompt = post_process(answer)

                        # post-process
                        if "Step 1." in base_long_prompt:
                            base_long_prompt = ""
                        if "Step 1." in negative_long_prompt:
                            negative_long_prompt = ""

                    base_long_output.append(base_long_prompt)
                    negative_long_output.append(negative_long_prompt)

                batch[sample_idx]['long_prompt'] = base_long_output
                batch[sample_idx]['negative_long_prompt'] = negative_long_output

                self.output_list.append(batch[sample_idx]) 
            
        except Exception as e:
            print(f"Error in test_step: {e}")
            traceback.print_exc()

            
    def prepare_input_embeds(self, pair_list): 
        # pair_list: list of Tuple (sub_category, base prompt, negative prompt)
        prepare_list = []
        valid_indices = []

        for idx, triplet in enumerate(pair_list):     
            if triplet is None:
                continue
            sub_category, base_prompt, negative_prompt = triplet
            get_func = get_prompt_dense[sub_category]
            system_prompt, conversation = get_func(base_prompt, negative_prompt)
            
            sft_format = get_sft_format(self.processor, system_prompt, conversation)
            prepare_list.append(get_processor_output(self.processor, self.tokenizer, sft_format))
            valid_indices.append(idx)

        if not prepare_list:
            print("No valid prompts found in the batch.")
            return [], []

        # batchify
        batched_prepares = batchify(self.processor, self.tokenizer, prepare_list)
        batched_prepares = batched_prepares.to(self.device) 

        inputs_embeds = self.model.prepare_inputs_embeds(**batched_prepares)

        # return inputs_embeds, batched_prepares
        return inputs_embeds, (batched_prepares, valid_indices)


    @torch.inference_mode()
    def generate(self, input_embeds, batched_prepares):
        outputs = self.model.language_model.generate(inputs_embeds=input_embeds,
            attention_mask=batched_prepares.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            **self.config.generation_config
            )

        return outputs


    def on_test_epoch_end(self):
        save_json_ddp(
            save_root=self.config.save_path,
            save_name='long_prompt' if self.config.save_name is None else self.config.save_name,
            world_size=self.trainer.world_size,
            save_file=self.output_list,
            rank=self.trainer.global_rank,
        )
        # print("Densification done.")
        print(f"Long prompt saved at {os.path.join(self.config.save_path, 'long_prompt.json')}")

