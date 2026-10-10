# Adapted from: https://github.com/LowinLi/transformers-stream-generator

import inspect
import random
import warnings
from typing import Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from transformers import GenerationConfig, GenerationMixin, LogitsProcessorList, PreTrainedModel, StoppingCriteriaList
from transformers.generation.utils import GenerateOutput, logger


def setup_seed(seed):
    if seed == -1:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class StreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_stream = kwargs.pop("do_stream", False)


class NewGenerationMixin(GenerationMixin):
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = False,
        seed=0,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for decoder-only models with a language modeling head, yielding one token at
        a time together with the final hidden state that produced it.

        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation. If `None` the method initializes it with
                `bos_token_id` and a batch size of 1.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the model's default generation configuration is used.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the generation to allowed tokens only at each step.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            kwargs:
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model.

        Return:
            A generator yielding `(next_tokens, latents)` tuples, see `sample_stream`.
        """
        # setup_seed(seed)
        # The legacy `do_stream` switch is accepted for compatibility; this method always streams.
        kwargs.pop("do_stream", None)
        if self.config.is_encoder_decoder:
            raise NotImplementedError("Streaming generation only supports decoder-only models.")

        # 1. Handle kwargs, `generation_config`, validate them and obtain generation mode
        generation_mode_kwargs = self._extract_generation_mode_kwargs(None, kwargs, synced_gpus, None, None)
        has_default_max_length = (
            kwargs.get("max_length") is None
            and (generation_config is None or generation_config.max_length is None)
            and self.generation_config.max_length is None
        )
        has_default_min_length = (
            kwargs.get("min_length") is None
            and (generation_config is None or generation_config.min_length is None)
            and self.generation_config.min_length is None
        )
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        generation_mode = generation_config.get_generation_mode()
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device, batch_size=batch_size)

        # decoder-only models must use left-padding for batched generation.
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

        # 4. Define other model kwargs
        if model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=False,
            **model_kwargs,
        )

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[1]
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        max_cache_length = generation_config.max_length - 1
        if inputs_tensor.shape[1] != input_ids_length and model_input_name == "inputs_embeds":
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=generation_mode_kwargs.get("tokenizer"),
        )

        # 9. forward the output flags to the model so the stream can read the hidden states
        model_kwargs["use_cache"] = generation_config.use_cache
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states

        # 10. run sample
        return self.sample_stream(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=generation_mode_kwargs.get("synced_gpus", False),
            **model_kwargs,
        )

    @torch.no_grad()
    def sample_stream(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling**
        (or greedy decoding when `do_sample=False`) and yields each step as it is produced.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model.

        Return:
            Generator that yields `(next_tokens, latents)` where `next_tokens` has shape `(batch_size,)` and
            `latents` is the normalized last hidden state of shape `(batch_size, hidden_size)`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_scores = generation_config.output_scores
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # the stopping criteria may read the scores, which are only collected when the caller asked for them
        scores = () if (return_dict_in_generate and output_scores) else None

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        if pad_token_id is not None:
            pad_token_id = pad_token_id.to(input_ids.device)

        is_first_iteration = True
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # the first pass feeds the whole prompt, later passes feed only the newest token when caching
            next_sequence_length = None if (is_first_iteration or not model_kwargs["use_cache"]) else 1
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                next_sequence_length=next_sequence_length,
                is_first_iteration=is_first_iteration,
                **model_kwargs,
            )
            is_first_iteration = False

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # the copy drops the reference to the full logits, which are large on the first iteration
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate and output_scores:
                scores += (next_token_scores,)

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            yield next_tokens, self.final_norm(outputs.hidden_states[-1][:, -1])

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            del outputs


def init_stream_support():
    """Overload PreTrainedModel for streaming."""
    PreTrainedModel.generate_stream = NewGenerationMixin.generate
    PreTrainedModel.sample_stream = NewGenerationMixin.sample_stream
