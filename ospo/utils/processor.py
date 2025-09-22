
import torch
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from janus.models.processing_vlm import VLChatProcessorOutput, BatchedVLChatProcessorOutput


def get_conversation(prompt: str):
    conversation = [
        {
            "role": "User",
            "content": prompt, # "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
        },
        {"role": "Assistant", "content": ""},
    ]

    return conversation


def get_sft_format(processor, system_prompt, conversation: list):
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format='deepseek',
                system_prompt=system_prompt
            )
    
    return sft_format


def get_processor_output(processor, tokenizer, sft_format):
    image_id = tokenizer.vocab.get("<image_placeholder>")
    image_token_num_per_image = 576

    input_ids = processor.tokenizer.encode(sft_format)
    input_ids = torch.LongTensor(input_ids)    

    # without image
    image_token_mask = input_ids == image_id
    image_indices = image_token_mask.nonzero()

    images_outputs = processor.image_processor([], return_tensors="pt")
    num_image_tokens = torch.IntTensor([image_token_num_per_image] * len(image_indices))

    output = VLChatProcessorOutput( # = prepare
        sft_format=sft_format,
        input_ids=input_ids,
        pixel_values=images_outputs.pixel_values,
        num_image_tokens=num_image_tokens,
    )

    return output


def batchify(processor, tokenizer, processor_outputs: list): 
    batch_size = len(processor_outputs)
    pad_id = tokenizer.vocab.get("<｜▁pad▁｜>")
    image_id = tokenizer.vocab.get("<image_placeholder>")
    image_token_num_per_image = 576

    seq_lengths = [len(item) for item in processor_outputs]
    image_lengths = [len(item.num_image_tokens) for item in processor_outputs]

    max_seq_length = max(seq_lengths)       
    max_image_length = max(1, max(image_lengths)) 

    batched_input_ids = torch.full((batch_size, max_seq_length), pad_id).long()  # FIXME
    batched_attention_mask = torch.zeros((batch_size, max_seq_length)).long()
    batched_pixel_values = torch.zeros((batch_size, max_image_length, *processor.image_processor.default_shape)).float()
    batched_images_seq_mask = torch.zeros((batch_size, max_seq_length)).bool()
    batched_images_emb_mask = torch.zeros((batch_size, max_image_length, image_token_num_per_image)).bool()

    batched_sft_format = []
    for i, item in enumerate(processor_outputs): 
        input_ids = item.input_ids
        seq_len = seq_lengths[i] 
        image_len = image_lengths[i] # number of image in single sample
                              
        # left-padding
        batched_attention_mask[i, -seq_len:] = 1
        batched_input_ids[i, -seq_len:] = torch.LongTensor(input_ids)
        batched_images_seq_mask[i, -seq_len:] = input_ids == image_id

        if image_len > 0:
            batched_pixel_values[i, :image_len] = item.pixel_values
            for j, num_image_tokens in enumerate(image_len):
                batched_images_emb_mask[i, j, :num_image_tokens] = True

        batched_sft_format.append(item.sft_format)

    batched_output = BatchedVLChatProcessorOutput(
        input_ids=batched_input_ids,
        attention_mask=batched_attention_mask,
        pixel_values=batched_pixel_values,
        images_seq_mask=batched_images_seq_mask,
        images_emb_mask=batched_images_emb_mask,
        sft_format=batched_sft_format,
    )

    return batched_output
  