#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Yanwei Li
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union

from transformers.utils import logging
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from model.arhead import AR_head
from model.liquid import MiniGeminiMetaModel, MiniGeminiMetaForCausalLM


logger = logging.get_logger(__name__)


class MiniGeminiConfig(LlamaConfig):
    model_type = "mini_gemini"


class MiniGeminiLlamaModel(MiniGeminiMetaModel, LlamaModel):
    config_class = MiniGeminiConfig

    def __init__(self, config: LlamaConfig):
        super(MiniGeminiLlamaModel, self).__init__(config)


class MiniGeminiLlamaForCausalLM(LlamaForCausalLM, MiniGeminiMetaForCausalLM):
    config_class = MiniGeminiConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MiniGeminiLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.ar_head = AR_head(self.config, codebook_size=32768, num_codebooks=8)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        data_types: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        additional_image_indexs = None
        if inputs_embeds is None and past_key_values is None:  # no in inference
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds, 
                labels,
                data_types,
                additional_image_labels,
                additional_image_indexs
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                images_aux,
                data_types
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        text_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            text_loss = loss_fct(shift_logits, shift_labels)
            num_text_tokens = (shift_labels != -100).sum().item()

            if num_text_tokens == 0: # 추가
                text_loss = torch.tensor(0.0, device=shift_logits.device)

        if additional_image_indexs is None:
            return CausalLMOutputWithPast(
                loss=text_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        to_image_mask = data_types == 1  # where to get t2i loss in each batch  [True, False, False, True....]

        if len(additional_image_indexs) > 0 and len(to_image_mask) == len(hidden_states):  # image generation loss
            to_image_states = hidden_states[to_image_mask] 

            # assert len(to_image_states) == len(additional_image_indexs)
            if len(to_image_states) != len(additional_image_indexs):
                print('to_image_mask', to_image_mask)
                print('additional_image_indexs', additional_image_indexs)
            shift_image_states = torch.stack([state[start_id - 1:end_id - 1] for (start_id, end_id), state in
                                              zip(additional_image_indexs, to_image_states)])  # Shift so that tokens < n predict n  [bz, seq_len, hidden_dim]
            base_tokens = shift_image_states # input token의 base

            K = self.ar_head.num_codebooks
            B, L, C = base_tokens.shape
            base_tokens = base_tokens.reshape(B * L, 1, C)

            targets = torch.cat(additional_image_labels, dim=0)  # [B, K, L]
            image_code_labels = targets
            targets = targets.permute(0, 2, 1).reshape(B * L, K)[:, :-1] # input token의 마지막 token을 제외한 7개 토큰
            index_embeddings = []
            for i in range(K - 1):
                index_embed = self.ar_head.codebooks[i](targets[:, i])
                index_embeddings.append(index_embed)
            index_embeddings = torch.stack(index_embeddings, dim=1)
            # import pdb;pdb.set_trace()
            h = torch.cat((base_tokens, index_embeddings), dim=1)  # [B*L, K, C] // (256 * B, 8, 4096)

            multicode_embedding = self.ar_head(
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=h,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
                cache_position=None,
            )
            image_logits = self.ar_head.linear_head(multicode_embedding[0])
            image_logits = image_logits.reshape(B, L, K, -1).permute(0, 2, 1, 3)  # [B, K, L, sub_vocab_size]
            loss_fct = CrossEntropyLoss()
            image_logits = image_logits.reshape(-1, self.ar_head.sub_vocab_size)
            image_labels = image_code_labels.view(-1)
            image_labels = image_labels.to(image_logits.device)
            image_softmax_normalizer = image_logits.max(-1).values ** 2
            image_z_loss = 0.00005 * image_softmax_normalizer.mean()
            image_loss = loss_fct(image_logits, image_labels) + image_z_loss
            num_image_tokens = image_labels.shape[0]
        else:
            if len(hidden_states) != len(to_image_mask):
                print('to_image_mask', to_image_mask)
                print('hidden_states', hidden_states.shape)
                print('inputs_embeds', inputs_embeds.shape)
                print('additional_image_indexs', additional_image_indexs)
            fake_ids = torch.ones(1, self.model.multi_embedder.num_codebooks - 1).to(inputs_embeds).long()
            index_embeddings = []
            for i in range(self.model.multi_embedder.num_codebooks - 1):
                index_embed = self.ar_head.codebooks[i](fake_ids[:, i])
                index_embeddings.append(index_embed)
            index_embeddings = torch.stack(index_embeddings, dim=1)

            multicode_embedding = self.ar_head(
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=index_embeddings,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
                cache_position=None,
            )
            image_logits = self.ar_head.linear_head(multicode_embedding[0])

            num_image_tokens = 0
            image_loss = (image_logits * 0).sum()  # + (base_tokens*0).sum()
            pass

        loss = image_loss * (num_image_tokens / (num_image_tokens + num_text_tokens)) + \
                text_loss * (num_text_tokens / (num_image_tokens + num_text_tokens))

        # t2i_ratio = to_image_mask.sum() / len(to_image_mask)
        # loss = image_loss * t2i_ratio + text_loss * (1 - t2i_ratio)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate_mllm(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        # import pdb;pdb.set_trace()
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                images_aux
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        # import pdb;pdb.set_trace()
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            images_aux: Optional[torch.FloatTensor] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                images_aux
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def test_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        input_multi_ids: torch.LongTensor = None,
        data_types: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # import pdb;pdb.set_trace()
        if input_multi_ids is not None:
            input_multi_ids = input_multi_ids.unsqueeze(-1)  # [B,K,1]
            input_ids = None  # [B,1]
            inputs_embeds = self.model.multi_embedder(input_multi_ids)  # [B,1,C]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    def T2I_forward_nocache(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        input_multi_ids: torch.LongTensor = None,
        data_types: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # import pdb;pdb.set_trace()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_multi_ids is not None:
            inputs_text_embeds = self.get_model().embed_tokens(input_ids)
            input_ids = None  # [B,1]
            inputs_image_embeds = self.model.multi_embedder(input_multi_ids)  # [B,1,C]
            inputs_image_mask = torch.empty(inputs_image_embeds.shape[0], inputs_image_embeds.shape[1]).fill_(1).to(
                attention_mask)
            inputs_embeds = torch.cat([inputs_text_embeds, inputs_image_embeds], dim=1)
            attention_mask = torch.cat([attention_mask, inputs_image_mask], dim=1)
            position_ids = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0).repeat(
                inputs_embeds.shape[0], 1)
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            input_ids = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    def T2I_forward_withcache(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        input_multi_ids: torch.LongTensor = None,
        data_types: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # import pdb;pdb.set_trace()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_multi_ids is not None:
            inputs_image_embeds = self.model.multi_embedder(input_multi_ids[:, :, -1:])  # [B,1,C]
            inputs_embeds = inputs_image_embeds
            input_ids = None  # [B,1]
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            input_ids = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        images_aux = kwargs.pop("images_aux", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if images_aux is not None:
            _inputs['images_aux'] = images_aux
        return _inputs


AutoConfig.register("mini_gemini", MiniGeminiConfig)
AutoModelForCausalLM.register(MiniGeminiConfig, MiniGeminiLlamaForCausalLM)