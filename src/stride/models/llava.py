from __future__ import annotations

import time
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ..config import RouterConfig
from ..math_utils import infer_grid
from ..progress import timed_stage
from ..routing import needs_semantic_alignment, needs_vision_attention, route as route_tokens
from ..types import RoutingContext
from .base import GenerationOutput, VLMAdapter
from .language import first_layer_value_space
from .vision import vision_routing_tensors


_CONTENT_STOPWORDS = frozenset(
    '''a an and are as at be been being both by can could did do does for from
    had has have he her hers him his how i if in into is it its me my of on or
    our ours she should that the their theirs them then there these they this
    those to was we were what when where which who why will with would you your
    yours image photo picture shown show see question answer exactly only word
    phrase explain'''.split()
)


class LlavaAdapter(VLMAdapter):
    '''Hugging Face LLaVA adapter with vision-language interface routing.'''

    def __init__(
        self,
        model_id: str,
        router_config: RouterConfig,
        device: str = 'cuda',
        dtype: str = 'bfloat16',
        attn_implementation: str | None = None,
        load_in_4bit: bool = False,
    ) -> None:
        try:
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                LlavaForConditionalGeneration,
            )
        except ImportError as exc:
            raise ImportError(
                'Install model dependencies with: pip install -e .[vlm]'
            ) from exc
        torch_dtype = getattr(torch, dtype)
        kwargs: dict[str, Any] = {
            'torch_dtype': torch_dtype,
            'low_cpu_mem_usage': True,
        }
        if attn_implementation:
            kwargs['attn_implementation'] = attn_implementation
        if load_in_4bit:
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs['device_map'] = 'auto'
        with timed_stage(f'load model {model_id}'):
            model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
            self.model = model if load_in_4bit else model.to(device)
            self.model.eval()
        with timed_stage(f'load processor {model_id}'):
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.device = torch.device(device)
        self.router_config = router_config

    def _routing_text_embeddings(
        self, prompt: str | None
    ) -> tuple[torch.Tensor | None, str]:
        """Embed content words only, without running the language decoder.

        Word-level averaging avoids giving multi-piece words extra weight.  The
        fixed stop list removes answer-formatting and grammatical tokens that
        have little visual grounding.  This is deterministic and training-free.
        """
        if not prompt:
            return None, 'missing_prompt'
        words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", prompt.lower())
        words = [
            word for word in words
            if word not in _CONTENT_STOPWORDS and (len(word) > 1 or word.isdigit())
        ]
        vectors = []
        embedding = self.model.get_input_embeddings()
        for word in words:
            ids = self.processor.tokenizer(
                word, add_special_tokens=False, return_tensors='pt'
            )['input_ids'].to(self.device)
            if ids.numel():
                vectors.append(embedding(ids)[0].float().mean(dim=0))
        if not vectors:
            return None, 'no_content_words'
        return torch.stack(vectors), 'content_word_input_embeddings'

    @torch.inference_mode()
    def generate(
        self,
        image: str | Path,
        prompt: str,
        method: str = 'stride',
        budget: int = 64,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        routing_prompt: str | None = None,
        **generation_kwargs: Any,
    ) -> GenerationOutput:
        request_start = time.perf_counter()
        conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            }
        ]
        if getattr(self.processor, 'chat_template', None):
            rendered = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
        else:
            rendered = f'USER: <image>\n{prompt}\nASSISTANT:'
        with Image.open(image) as source:
            inputs = self.processor(
                images=source.convert('RGB'),
                text=rendered,
                return_tensors='pt',
            )
        inputs = {
            key: value.to(self.device) if hasattr(value, 'to') else value
            for key, value in inputs.items()
        }
        input_ids = inputs['input_ids']
        image_token_id = self.model.config.image_token_index
        safe_ids = input_ids.masked_fill(input_ids == image_token_id, 0)
        text_embeddings = self.model.get_input_embeddings()(safe_ids)[0]
        self._sync()
        preprocessing_seconds = time.perf_counter() - request_start
        start = time.perf_counter()
        feature_layer = getattr(self.model.config, 'vision_feature_layer', -2)
        select_strategy = getattr(
            self.model.config, 'vision_feature_select_strategy', 'default'
        )
        global_feature = None
        if needs_vision_attention(method):
            (
                visual,
                vision_features,
                salience,
                global_feature,
                salience_source,
            ) = vision_routing_tensors(
                self.model.vision_tower,
                inputs['pixel_values'],
                feature_layer,
                select_strategy == 'default',
                self.model.multi_modal_projector,
            )
        else:
            image_features = self.model.get_image_features(
                inputs['pixel_values'],
                vision_feature_layer=feature_layer,
                vision_feature_select_strategy=select_strategy,
            )
            if hasattr(image_features, 'last_hidden_state'):
                image_features = image_features.last_hidden_state
            if isinstance(image_features, (tuple, list)):
                image_features = image_features[0]
            visual = image_features[0]
            vision_features = visual
            salience = torch.ones(len(visual), device=visual.device)
            salience_source = 'not_requested'
        self._sync()
        vision_seconds = time.perf_counter() - start

        grid = infer_grid(len(visual))
        start = time.perf_counter()
        config = RouterConfig.from_dict(
            {**self.router_config.to_dict(), 'budget': budget}
        )
        routing_text, text_source = None, 'not_requested'
        semantic_visual = None
        alignment_source = 'not_requested'
        if needs_semantic_alignment(method) and (
            config.stride_use_semantics or config.stride_use_residual_space
        ):
            routing_text, text_source = self._routing_text_embeddings(routing_prompt)
            semantic_visual, routing_text, alignment_source = first_layer_value_space(
                self.model, visual, routing_text
            )
        routed = route_tokens(
            method,
            RoutingContext(
                tokens=visual,
                vision_features=vision_features,
                vision_salience=salience,
                text_tokens=routing_text,
                semantic_visual_tokens=semantic_visual,
                text_source=text_source,
                routing_prompt=routing_prompt or '',
                language_relevance=None,
                alignment_source=alignment_source,
                grid=grid,
                salience_source=salience_source,
                global_vision_feature=global_feature,
                projector=self.model.multi_modal_projector,
            ),
            config,
            seed=self.router_config.seed,
        )
        self._sync()
        routing_seconds = time.perf_counter() - start

        start = time.perf_counter()
        packed = self._replace_image_span(
            text_embeddings, input_ids[0], image_token_id, routed.tokens
        )
        position_arguments = {}
        attention_mask = torch.ones(
            (1, packed.shape[0]), device=self.device, dtype=torch.long
        )
        self._sync()
        packing_seconds = time.perf_counter() - start
        prefill_seconds = (
            preprocessing_seconds + vision_seconds + routing_seconds + packing_seconds
        )
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        generation_kwargs.setdefault(
            'pad_token_id',
            self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id is not None
            else self.processor.tokenizer.eos_token_id,
        )
        generated = self.model.language_model.generate(
            inputs_embeds=packed.unsqueeze(0),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **position_arguments,
            **generation_kwargs,
        )
        self._sync()
        generation_seconds = time.perf_counter() - start
        text = self.processor.tokenizer.decode(
            generated[0], skip_special_tokens=True
        ).strip()
        peak = (
            torch.cuda.max_memory_allocated(self.device)
            if self.device.type == 'cuda'
            else None
        )
        route_diagnostics = dict(routed.diagnostics)
        route_diagnostics['input_grid'] = list(grid)
        route_diagnostics['dense_positions_preserved'] = False
        if routed.selected_indices is not None:
            route_diagnostics['selected_indices'] = (
                routed.selected_indices.detach().cpu().tolist()
            )
        return GenerationOutput(
            text=text,
            input_visual_tokens=len(visual),
            output_visual_tokens=len(routed.tokens),
            prefill_seconds=prefill_seconds,
            generation_seconds=generation_seconds,
            peak_memory_bytes=peak,
            route_diagnostics=route_diagnostics,
            preprocessing_seconds=preprocessing_seconds,
            vision_seconds=vision_seconds,
            routing_seconds=routing_seconds,
            packing_seconds=packing_seconds,
            total_seconds=time.perf_counter() - request_start,
        )

    @staticmethod
    def _replace_image_span(
        text_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        image_token_id: int,
        visual: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.nonzero(input_ids == image_token_id, as_tuple=False).flatten()
        if len(positions) == 0:
            raise ValueError('processor output contains no image placeholder tokens')
        first, last = int(positions[0]), int(positions[-1])
        if len(positions) != last - first + 1:
            raise NotImplementedError('only one contiguous image span is supported')
        return torch.cat(
            (text_embeddings[:first], visual, text_embeddings[last + 1 :]), dim=0
        )

    def _sync(self) -> None:
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)


MODEL_REGISTRY = {
    'llava15-7b': 'llava-hf/llava-1.5-7b-hf',
    'llava15-13b': 'llava-hf/llava-1.5-13b-hf',
}
