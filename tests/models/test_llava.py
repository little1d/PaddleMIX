# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import unittest
import numpy as np
import paddle


# 配置和模型定义的导入
from paddlemix.models.llava.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
from paddlemix.models.llava.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from paddlemix.models.llava.language_model.tokenizer import LLavaTokenizer

# 测试工具导入
from tests.models.test_configuration_common import ConfigTester
from tests.models.test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from tests.testing_utils import slow


class LlavaModelTester:
    def __init__(self, parent, model_name):
        self.parent = parent
        self.model_name = model_name
        # TODO
        # self.tokenizer = LLavaTokenizer.from_pretrained("path_to_tokenizer")

    def get_config(self):
        # llava_qwen config copy from https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov/blob/main/config.json
        test_llava_qwen_config = {
            "_name_or_path": "lmms-lab/llava-onevision-qwen2-7b-ov",
            "architectures": ["LlavaQwenForCausalLM"],
            "mm_newline_position": "one_token",
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 3584,
            "image_token_index": 151646,
            "image_aspect_ratio": "anyres_max_9",
            "image_crop_resolution": None,
            "image_grid_pinpoints": [
                [384, 384],
                [384, 768],
                [384, 1152],
                [384, 1536],
                [384, 1920],
                [384, 2304],
                [768, 384],
                [768, 768],
                [768, 1152],
                [768, 1536],
                [768, 1920],
                [768, 2304],
                [1152, 384],
                [1152, 768],
                [1152, 1152],
                [1152, 1536],
                [1152, 1920],
                [1152, 2304],
                [1536, 384],
                [1536, 768],
                [1536, 1152],
                [1536, 1536],
                [1536, 1920],
                [1536, 2304],
                [1920, 384],
                [1920, 768],
                [1920, 1152],
                [1920, 1536],
                [1920, 1920],
                [1920, 2304],
                [2304, 384],
                [2304, 768],
                [2304, 1152],
                [2304, 1536],
                [2304, 1920],
                [2304, 2304],
            ],
            "image_split_resolution": None,
            "initializer_range": 0.02,
            "intermediate_size": 18944,
            "max_position_embeddings": 32768,
            "max_window_layers": 28,
            "mm_hidden_size": 1152,
            "mm_patch_merge_type": "spatial_unpad",
            "mm_projector_lr": None,
            "mm_projector_type": "mlp2x_gelu",
            "mm_resampler_type": None,
            "mm_spatial_pool_mode": "bilinear",
            "mm_tunable_parts": "mm_vision_tower,mm_mlp_adapter,mm_language_model",
            "mm_use_im_patch_token": False,
            "mm_use_im_start_end": False,
            "mm_vision_select_feature": "patch",
            "mm_vision_select_layer": -2,
            "mm_vision_tower": "google/siglip-so400m-patch14-384",
            "mm_vision_tower_lr": 2e-06,
            "model_type": "llava",
            "num_attention_heads": 28,
            "num_hidden_layers": 28,
            "num_key_value_heads": 4,
            "pos_skipping_range": 4096,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 1000000.0,
            "sliding_window": 131072,
            "tie_word_embeddings": False,
            "tokenizer_model_max_length": 32768,
            "tokenizer_padding_side": "right",
            "torch_dtype": "bfloat16",
            "transformers_version": "4.40.0.dev0",
            "use_cache": True,
            "use_mm_proj": True,
            "use_pos_skipping": False,
            "use_sliding_window": False,
            "vision_tower_pretrained": None,
            "vocab_size": 152064,
        }
        # llava_llama configs copy from https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b/blob/main/config.json
        test_llava_llama_config = {
            "_name_or_path": "liuhaotian/llava-v1.6-vicuna-7b",
            "architectures": ["LlavaLlamaForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "freeze_mm_mlp_adapter": False,
            "freeze_mm_vision_resampler": False,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "image_aspect_ratio": "anyres",
            "image_crop_resolution": 224,
            "image_grid_pinpoints": [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]],
            "image_split_resolution": 224,
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "mm_hidden_size": 1024,
            "mm_patch_merge_type": "spatial_unpad",
            "mm_projector_lr": None,
            "mm_projector_type": "mlp2x_gelu",
            "mm_resampler_type": None,
            "mm_use_im_patch_token": False,
            "mm_use_im_start_end": False,
            "mm_vision_select_feature": "patch",
            "mm_vision_select_layer": -2,
            "mm_vision_tower": "openai/clip-vit-large-patch14-336",
            "mm_vision_tower_lr": 2e-06,
            "model_type": "llava",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "pad_token_id": 0,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "tokenizer_model_max_length": 4096,
            "tokenizer_padding_side": "right",
            "torch_dtype": "bfloat16",
            "transformers_version": "4.36.2",
            "tune_mm_mlp_adapter": False,
            "tune_mm_vision_resampler": False,
            "unfreeze_mm_vision_tower": True,
            "use_cache": True,
            "use_mm_proj": True,
            "vocab_size": 32000,
        }
        if self.model_name == "LlavaQwen":
            return LlavaQwenConfig(**test_llava_qwen_config)
        elif self.model_name == "LlavaLlama":
            return LlavaConfig(**test_llava_llama_config)

    def prepare_config_and_inputs(self):
        # inputs
        images = ([floats_tensor([3, 224, 224])],)
        tokenized_out = {
            "input_ids": ids_tensor([1, 258], 5000),
            "token_type_ids": random_attention_mask([1, 258]),
            "attention_mask": random_attention_mask([1, 258]),
            "position_ids": ids_tensor([1, 258], vocab_size=100),
        }
        # config
        config = self.get_config()
        return config, images, tokenized_out

    def prepare_config_and_inputs_for_common(self):
        config, images, tokenized_out = self.prepare_config_and_inputs()
        inputs_dict = {
            "images": images,
            "input_ids": tokenized_out["input_ids"],
            "attention_mask": tokenized_out["attention_mask"],
            "token_type_ids": tokenized_out["token_type_ids"],
            "position_id": tokenized_out["position_ids"],
        }

        return config, inputs_dict

    def create_and_check_model(self, model_class, images, input_ids, attention_mask, token_type_ids, position_id):
        model = model_class(config=self.get_config())
        model.eval()
        with paddle.no_grad():
            result = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_id=position_id,
            )
        self.parent.assertIsNotNone(result)


# factory function to create LlavaModelTester instance
def create_llava_model_tester(parent, model_name):
    if model_name == "LlavaQwen":
        return LlavaModelTester(parent, model_name)
    elif model_name == "LlavaLlama":
        return LlavaModelTester(parent, model_name)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


class LlavaModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (LlavaQwenForCausalLM, LlavaLlamaForCausalLM)
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False
    use_test_inputs_embeds: bool = False

    def setUp(self):
        # model tester instance
        self.LlavaQwen_model_tester = create_llava_model_tester(self, "LlavaQwen")
        self.LlavaLlama_model_tester = create_llava_model_tester(self, "LlavaLlama")

        # config tester instance
        self.LlavaQwen_config_tester = ConfigTester(
            self,
            config_class=LlavaQwenConfig,
        )
        self.LlavaLlama_config_tester = ConfigTester(
            self,
            config_class=LlavaConfig,
        )

    def test_config(self):
        self.LlavaQwen_config_tester.run_common_tests()
        self.LlavaLlama_config_tester.run_common_tests()

    def test_determinism(self):
        def check_determinism(first, second):
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 5e-5)

        for model_class in self.all_model_classes:
            if model_class == LlavaQwenForCausalLM:
                tester = self.LlavaQwen_model_tester
            elif model_class == LlavaLlamaForCausalLM:
                tester = self.LlavaLlama_model_tester
            config, inputs_dict = tester.prepare_config_and_inputs()
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                first = model(**inputs_dict)
                second = model(**inputs_dict)
            check_determinism(first, second)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass
    def test_model(self):
        for model_class in self.all_model_classes:
            if model_class == LlavaQwenForCausalLM:
                tester = self.LlavaQwen_model_tester
            elif model_class == LlavaLlamaForCausalLM:
                tester = self.LlavaLlama_model_tester
            config, inputs_dict = tester.prepare_config_and_inputs_for_common()
            tester.create_and_check_model(model_class, **inputs_dict)

    # @slow
    # def test_model_from_pretrained(self):
    #     for model_class in self.all_model_classes:
    #         # TODO
    #         model = model_class.from_pretrained("path_to_pretrained_model")
    #         self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
