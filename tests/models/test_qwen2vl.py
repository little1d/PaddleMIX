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
os.environ["FLAGS_use_cuda_managed_memory"] = "true"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import unittest
import numpy as np
import paddle


# 配置和模型定义的导入
from paddlemix import QwenVLProcessor, QWenVLTokenizer
from paddlemix.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

# 测试工具导入
from tests.models.test_configuration_common import ConfigTester
from tests.models.test_modeling_common import (
    ModelTesterMixin,
)
from tests.testing_utils import (
    slow,
)

class Qwen2VLModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.model_name_or_path = "Qwen/Qwen2-VL-7B-Instruct"
        self.tokenizer = QWenVLTokenizer.from_pretrained(self.model_name_or_path)
        self.processor = QwenVLProcessor(tokenizer=self.tokenizer)

    def get_config(self):
        # configs copy from https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/blob/main/config.json
        test_config = {
            "architectures": [
                "Qwen2VLForConditionalGeneration"
            ],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
            "vision_token_id": 151654,
            "image_token_id": 151655,
            "video_token_id": 151656,
            "hidden_act": "silu",
            "hidden_size": 3584,
            "initializer_range": 0.02,
            "intermediate_size": 18944,
            "max_position_embeddings": 32768,
            "max_window_layers": 28,
            "model_type": "qwen2_vl",
            "num_attention_heads": 28,
            "num_hidden_layers": 28,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "sliding_window": 32768,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.41.2",
            "use_cache": True,
            "use_sliding_window": False,
            "vision_config": {
                "depth": 32,
                "embed_dim": 1280,
                "mlp_ratio": 4,
                "num_heads": 16,
                "in_chans": 3,
                "hidden_size": 3584,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "spatial_patch_size": 14,
                "temporal_patch_size": 2
            },
            "rope_scaling": {
                "type": "mrope",
                "mrope_section": [
                16,
                24,
                24
                ]
            },
            "vocab_size": 152064
            }
        return  Qwen2VLConfig(**test_config)
    
    def prepare_config_and_inputs(self):
        query = []
        query.append({"image": "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"})
        query.append({"text": "Generate the caption in English with grounding:"})
        inputs = self.processor(query=query, return_tensors="pd")
        config = self.get_config()

        return config, inputs
    
    def prepare_config_and_inputs_for_common(self):
        config, inputs = self.prepare_config_and_inputs()
        return config, inputs

    def create_and_check_model(self, kwargs):
        model = Qwen2VLModel(config=self.get_config())
        model.eval()
        with paddle.no_grad():
            result = model(**kwargs)

        self.parent.assertIsNotNone(result)

class Qwen2VLModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = ( Qwen2VLModel,)
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False
    use_test_inputs_embeds: bool = False
    def setUp(self):
        self.model_tester = Qwen2VLModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Qwen2VLConfig,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(inputs_dict)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = Qwen2VLModel.from_pretrained("qwen-vl/qwen-vl-chat-7b")
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
