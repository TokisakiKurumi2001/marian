from transformers import PreTrainedModel
from transformers.models.marian.modeling_marian import MarianEncoder, MarianDecoder
import torch
import torch.nn as nn
from typing import List, Union, Optional, Tuple, Dict
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from contextlib import contextmanager
from dataclasses import dataclass
from marian import MarianOTConfig

class MarianOTPreTrainedModel(PreTrainedModel):
    config_class = MarianOTConfig
    base_model_prefix = "marian_ot"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Embedding]):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (MarianDecoder, MarianEncoder)):
            module.gradient_checkpointing = value

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
        }
        return dummy_inputs

@dataclass
class CombineSeq2SeqLMOutput:
    tgt: Seq2SeqLMOutput
    tsl: Seq2SeqLMOutput

class MarianOT(MarianOTPreTrainedModel):
    def __init__(self, config: MarianOTConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # We always use self.shared for token embeddings to ensure compatibility with all marian models
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        if self.config.share_encoder_decoder_embeddings:
            encoder_embed_tokens = decoder_embed_tokens = self.shared
        else:
            # Since the embeddings are not shared, deepcopy the embeddings here for encoder
            # and decoder to make sure they are not tied.
            encoder_embed_tokens = copy.deepcopy(self.shared)
            decoder_embed_tokens = copy.deepcopy(self.shared)
            self.shared = None

        self.encoder = MarianEncoder(config, encoder_embed_tokens)
        self.tgt_decoder = MarianDecoder(config, decoder_embed_tokens)
        self.tgt_lm_head = nn.Linear(config.d_model, config.vocab_size, bias=True)
        self.tsl_decoder = MarianDecoder(config, decoder_embed_tokens)
        self.tsl_lm_head = nn.Linear(config.d_model, config.vocab_size, bias=True)
        
        self.translate_self_flag = False

        # Initialize weights and apply final processing
        self.post_init()

    @contextmanager
    def translate_self(self):
        self.translate_self_flag = True
        yield
        self.translate_self_flag = False

    def get_input_embeddings(self):
        # This will return shared embeddings if they are shared else specific to encoder.
        return self.get_encoder().get_input_embeddings()

    def set_input_embeddings(self, value):
        if self.config.share_encoder_decoder_embeddings:
            self.shared = value
            self.encoder.embed_tokens = self.shared
            self.tsl_decoder.embed_tokens = self.shared
            self.tgt_decoder.embed_tokens = self.shared
        else:  # if not shared only set encoder embeedings
            self.encoder.embed_tokens = value

    def get_decoder_input_embeddings(self):
        if self.config.share_encoder_decoder_embeddings:
            raise ValueError(
                "`get_decoder_input_embeddings` should not be called if `config.share_encoder_decoder_embeddings` "
                "is `True`. Please use `get_input_embeddings` instead."
            )
        return self.get_decoder().get_input_embeddings()

    def set_decoder_input_embeddings(self, value):
        if self.config.share_encoder_decoder_embeddings:
            raise ValueError(
                "`config.share_encoder_decoder_embeddings` is set to `True` meaning the decoder input embeddings "
                "are shared with the encoder. In order to set the decoder input embeddings, you should simply set "
                "the encoder input embeddings by calling `set_input_embeddings` with the appropriate embeddings."
            )
        self.tsl_decoder.embed_tokens = value
        self.tgt_decoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        if self.translate_self_flag:
            return self.tgt_decoder
        else:
            return self.tsl_decoder

    def resize_decoder_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        if self.config.share_encoder_decoder_embeddings:
            raise ValueError(
                "`resize_decoder_token_embeddings` should not be called if `config.share_encoder_decoder_embeddings` "
                "is `True`. Please use `resize_token_embeddings` instead."
            )

        old_embeddings = self.get_decoder_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_decoder_input_embeddings(new_embeddings)

        model_embeds = self.get_decoder_input_embeddings()

        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.decoder_vocab_size = new_num_tokens

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def forward_pass(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        tsl_decoder_input_ids: Optional[torch.LongTensor] = None,
        tgt_decoder_input_ids: Optional[torch.LongTensor] = None,
        tsl_decoder_attention_mask: Optional[torch.Tensor] = None,
        tgt_decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        tsl_decoder_head_mask: Optional[torch.Tensor] = None,
        tgt_decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CombineSeq2SeqLMOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        decoder_outputs_tsl = self.tsl_decoder(
            input_ids=tsl_decoder_input_ids,
            attention_mask=tsl_decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=tsl_decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        decoder_outputs_tgt = self.tgt_decoder(
            input_ids=tgt_decoder_input_ids,
            attention_mask=tgt_decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=tgt_decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        tsl_outputs = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs_tsl.last_hidden_state,
            past_key_values=decoder_outputs_tsl.past_key_values,
            decoder_hidden_states=decoder_outputs_tsl.hidden_states,
            decoder_attentions=decoder_outputs_tsl.attentions,
            cross_attentions=decoder_outputs_tsl.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        tgt_outputs = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs_tgt.last_hidden_state,
            past_key_values=decoder_outputs_tgt.past_key_values,
            decoder_hidden_states=decoder_outputs_tgt.hidden_states,
            decoder_attentions=decoder_outputs_tgt.attentions,
            cross_attentions=decoder_outputs_tgt.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        tgt_lm_logits = self.tgt_lm_head(tgt_outputs[0])
        tsl_lm_logits = self.tsl_lm_head(tsl_outputs[0])

        tgt_lm_output = Seq2SeqLMOutput(
            loss=None,
            logits=tgt_lm_logits,
            past_key_values=tgt_outputs.past_key_values,
            decoder_hidden_states=tgt_outputs.decoder_hidden_states,
            decoder_attentions=tgt_outputs.decoder_attentions,
            cross_attentions=tgt_outputs.cross_attentions,
            encoder_last_hidden_state=tgt_outputs.encoder_last_hidden_state,
            encoder_hidden_states=tgt_outputs.encoder_hidden_states,
            encoder_attentions=tgt_outputs.encoder_attentions,
        )

        tsl_lm_output = Seq2SeqLMOutput(
            loss=None,
            logits=tsl_lm_logits,
            past_key_values=tsl_outputs.past_key_values,
            decoder_hidden_states=tsl_outputs.decoder_hidden_states,
            decoder_attentions=tsl_outputs.decoder_attentions,
            cross_attentions=tsl_outputs.cross_attentions,
            encoder_last_hidden_state=tsl_outputs.encoder_last_hidden_state,
            encoder_hidden_states=tsl_outputs.encoder_hidden_states,
            encoder_attentions=tsl_outputs.encoder_attentions,
        )

        return CombineSeq2SeqLMOutput(
            tgt=tgt_lm_output,
            tsl=tsl_lm_output
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Seq2SeqLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        if self.translate_self_flag:
            decoder_fn = self.tgt_decoder
            lm_fn = self.tgt_lm_head
        else:
            decoder_fn = self.tsl_decoder
            lm_fn = self.tsl_lm_head

        decoder_outputs = decoder_fn(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
        lm_logits = lm_fn(outputs[0])

        masked_lm_loss = None

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids: torch.LongTensor,
        past: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
        **kwargs,
    ) -> Dict:
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def adjust_logits_during_generation(self, logits, cur_len):
        logits[:, self.config.pad_token_id] = float("-inf")  # never predict pad token.
        return logits

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past