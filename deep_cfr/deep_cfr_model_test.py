# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
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

# Lint as: python3
# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
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
"""Tests for open_spiel.python.algorithms.alpha_zero.model."""

from absl.testing import absltest
from absl.testing import parameterized
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import deep_cfr_model as model_lib

NUM_PARALLEL_EXEC_UNITS = 8
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                        allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.Session(config=config)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"


def build_model(model_type):
    return model_lib.CFRModel.build_model(
        model_type, [1, 1, 8], 3,
        nn_width=64, nn_depth=1, weight_decay=1e-4, learning_rate=0.01, path=None, device="/cpu:0")


class ModelTest(absltest.TestCase):

    def test_value_model_learns_simple(self):
        model_type = 'normal'
        with tf.device("/cpu:0"):
            model = build_model(model_type)
        self.assertIsNotNone(model._value)
        self.assertIsNone(model._baseline)
        self.assertIsNone(model._policy)
        self.assertIsNone(model._baseline_targets)
        print("Num variables:", model.num_trainable_variables)
        model.print_trainable_variables()

        train_inputs = []
        for i in range(10000):
            act_mask = np.array([1, 1, 1], dtype=np.bool)
            obs = np.random.randint(0, 10, 10)
            target = obs[:3]
            target[np.logical_not(np.array(act_mask, dtype=np.bool))] = 0
            train_inputs.append(model_lib.TrainInput(
                obs, act_mask, target, weight=[1]))
            # value = model.inference([obs], [act_mask])[0]
        time_start = time.time()
        losses = []
        for i in range(100):
            loss = model.update(train_inputs, 0.001)
            print(i, loss)
            losses.append(loss)
            # if loss < 0.05:
            #     break

        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        print(10 * 10000 / (time_end - time_start), 'states/s')
        # self.assertGreater(losses[0], losses[-1])
        # self.assertLess(losses[-1], 0.05)

        # Create the Timeline object, and write it to a json
        # tl = timeline.Timeline(run_metadata.step_stats)
        # ctf = tl.generate_chrome_trace_format()
        # with open('timeline.json', 'w') as f:
        #     f.write(ctf)


if __name__ == "__main__":
    absltest.main()
