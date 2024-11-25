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
from tkinter.messagebox import NO

os.environ["FLAGS_use_cuda_managed_memory"] = "True"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import unittest
import numpy as np
import paddle

True
from paddlemix.models.llava.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM

from tests.models.test_configuration_common import ConfigTester
from tests.models.test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from tests.testing_utils import slow


class LlavaModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.model_name_or_path = "lmms-lab/llava-next-interleave-qwen-0.5b"

    def get_config(self):
        # llava_llama configs copy from paddlemix lmms-lab/llava-next-interleave-qwen-0.5b
        test_config = {
            "_name_or_path": "./",
            "architectures": ["LlavaQwenForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "image_aspect_ratio": "anyres",
            "image_crop_resolution": 224,
            "image_grid_pinpoints": [[384, 768], [768, 384], [768, 768], [1152, 384], [384, 1152]],
            "image_split_resolution": 224,
            "initializer_range": 0.02,
            "intermediate_size": 2816,
            "max_position_embeddings": 32768,
            "max_window_layers": 21,
            "mm_hidden_size": 1152,
            "mm_patch_merge_type": "spatial_unpad",
            "mm_projector_lr": None,
            "mm_projector_type": "mlp2x_gelu",
            "mm_resampler_type": None,
            "mm_tunable_parts": "mm_mlp_adapter,mm_language_model",
            "mm_use_im_patch_token": False,
            "mm_use_im_start_end": False,
            "mm_vision_select_feature": "patch",
            "mm_vision_select_layer": -2,
            "mm_vision_tower": "google/siglip-so400m-patch14-384",
            "mm_vision_tower_lr": None,
            "model_type": "llava_qwen",
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "num_key_value_heads": 16,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 1000000.0,
            "sliding_window": 32768,
            "tie_word_embeddings": True,
            "tokenizer_model_max_length": 32768,
            "tokenizer_padding_side": "right",
            "torch_dtype": "bfloat16",
            "transformers_version": "4.39.0.dev0",
            "use_cache": True,
            "use_mm_proj": True,
            "use_sliding_window": False,
            "vision_tower_pretrained": None,
            "vocab_size": 151936,
        }

        return LlavaQwenConfig(**test_config)

    def prepare_config_and_inputs(self):
        # inputs
        images = floats_tensor([1, 5, 3, 336, 336])
        tokenized_out = {
            "input_ids": ids_tensor([1, 50], 5000),
            "attention_mask": random_attention_mask([1, 50]),
            "image_size": [(640, 429)],
            "position_ids": ids_tensor([1, 50], vocab_size=100),
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
            "position_ids": tokenized_out["position_ids"],
            "image_size": tokenized_out["image_size"],
        }

        return config, inputs_dict

    def create_and_check_model(self, images, input_ids, image_size, attention_mask, position_ids):
        model = LlavaQwenForCausalLM(config=self.get_config())
        model.eval()
        with paddle.no_grad():
            result = model(
                images=images,
                input_ids=input_ids,
                image_size=image_size,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        self.parent.assertIsNotNone(result)


class LlavaModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (LlavaQwenForCausalLM,)
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False
    use_test_inputs_embeds: bool = False

    def setUp(self):
        # model tester instance
        self.model_tester = LlavaModelTester(self)

        self.config_tester = ConfigTester(
            self,
            config_class=LlavaQwenConfig,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 5e-5)

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                first = model(**inputs_dict)
                second = model(**inputs_dict)

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(**inputs_dict)

    @slow
    def test_model_from_pretrained(self):
        model = LlavaQwenForCausalLM.from_pretrained("..../")
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
