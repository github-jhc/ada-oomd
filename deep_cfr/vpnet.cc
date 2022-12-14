// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "vpnet.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/run_python.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorMap.h"

namespace tf = tensorflow;
using Tensor = Eigen::Tensor<float, 2, Eigen::RowMajor>;
using TensorMap = Eigen::TensorMap<Tensor, Eigen::Aligned>;
using TensorBool = Eigen::Tensor<bool, 2, Eigen::RowMajor>;
using TensorMapBool = Eigen::TensorMap<TensorBool, Eigen::Aligned>;

namespace open_spiel {
namespace algorithms {

std::ostream& operator<<(std::ostream& io, const CFRNetModel::TrainInputs& ti) {
  io << "[" << ti.legal_actions << " " << ti.informations << " " << ti.value
     << "]";
  return io;
}

std::vector<double> PolicyNormalize(const std::vector<double>& weights) {
  std::vector<double> probs(weights);
  absl::c_for_each(probs, [](double& w) { w = (w > 0 ? w : 0); });
  const double normalizer = absl::c_accumulate(probs, 0.);
  absl::c_for_each(probs, [&probs, normalizer](double& w) {
    w = (normalizer == 0.0 ? 1.0 / probs.size() : w / normalizer);
  });
  return probs;
}

std::ostream& operator<<(std::ostream& io,
                         const CFRNetModel::InferenceInputs& ti) {
  io << "[" << ti.legal_actions << " " << ti.informations << "]";
  return io;
}
std::ostream& operator<<(std::ostream& io,
                         const CFRNetModel::InferenceOutputs& ti) {
  io << "[" << ti.value << "]";
  return io;
}

template <typename T>
void no_null_delete(T* ptr) {
  if (ptr != nullptr) {
    delete ptr;
  }
}

bool CreateGraphDef(const Game& game, double learning_rate, double weight_decay,
                    const std::string& path, const std::string& device,
                    const std::string& filename, std::string nn_model,
                    int nn_width, int nn_depth, bool verbose) {
  // NOTE: we use the first dimention of tensor as tensor size.
  std::vector<std::string> parameters{
      "--num_actions",
      absl::StrCat(game.NumDistinctActions()),  //
      "--path",
      absl::StrCat("'", path, "'"),  //
      "--device",
      absl::StrCat("'", device, "'"),  //
      "--graph_def",
      filename,  //
      "--learning_rate",
      absl::StrCat(learning_rate),  //
      "--weight_decay",
      absl::StrCat(weight_decay),  //
      "--nn_model",
      nn_model,  //
      "--nn_depth",
      absl::StrCat(nn_depth),  //
      "--nn_width",
      absl::StrCat(nn_width),  //
      absl::StrCat("--verbose=", verbose ? "true" : "false"),
  };
  for (auto& ts : game.InformationStateTensorShape()) {
    parameters.insert(parameters.end(), {"--tensor_shape", absl::StrCat(ts)});
  }
  return RunPython("deep_cfr.export_vpnet", parameters);
}

void CreateModel(const Game& game, double learning_rate, double weight_decay,
                 const std::string& model_path,
                 const std::string& model_name_prefix,
                 std::string regret_nn_model, std::string policy_nn_model,
                 int nn_width, int nn_depth, bool use_gpu) {
  std::vector<std::string> model_names{"value_0", "value_1", "policy_0",
                                       "policy_1"};
  for (auto& model_name : model_names) {
    std::string nn_model = model_name.find("policy") == std::string::npos
                               ? regret_nn_model
                               : policy_nn_model;
    SPIEL_CHECK_TRUE(CreateGraphDef(
        game, learning_rate, weight_decay, model_path, "/cpu:0",
        absl::StrJoin({model_name_prefix, model_name, std::string("cpu.pb")},
                      "_"),
        nn_model, nn_width, nn_depth));
    if (use_gpu) {
      SPIEL_CHECK_TRUE(CreateGraphDef(
          game, learning_rate, weight_decay, model_path, "/gpu:0",
          absl::StrJoin({model_name_prefix, model_name, std::string("gpu.pb")},
                        "_"),
          nn_model, nn_width, nn_depth));
    }
  }
}

CFRNetModel::CFRNetModel(const Game& game, const std::string& path,
                         const std::string& model_path,
                         const std::string& file_name, int num_threads,
                         const std::string& device, bool use_value_net,
                         bool seperate_value_net, bool use_policy_net,
                         bool seperate_policy_net, bool use_target)
    : device_(device),
      path_(path),
      num_threads_(num_threads),
      use_value_net(use_value_net),
      use_policy_net(use_policy_net),
      seperate_value_net(seperate_value_net),
      seperate_policy_net(seperate_policy_net) {
  // Some assumptions that we can remove eventually. The value net returns
  // a single value in terms of player 0 and the game is assumed to be zero-sum,
  // so player 1 can just be -value.
  if (use_value_net) {
    value_net_0 = std::shared_ptr<CFRNet>(
        new CFRNet(game, path, model_path, file_name + "_value_0", num_threads_,
                   device, use_target));
    if (seperate_value_net) {
      value_net_1 = std::shared_ptr<CFRNet>(
          new CFRNet(game, path, model_path, file_name + "_value_1",
                     num_threads_, device, use_target));
    } else {
      value_net_1 = value_net_0;
    }
  }
  if (use_policy_net) {
    policy_net_0 = std::shared_ptr<CFRNet>(
        new CFRNet(game, path, model_path, file_name + "_policy_0",
                   num_threads_, device, use_target));
    if (seperate_policy_net) {
      policy_net_1 = std::shared_ptr<CFRNet>(
          new CFRNet(game, path, model_path, file_name + "_policy_1",
                     num_threads_, device, use_target));
    } else {
      policy_net_1 = policy_net_0;
    }
  }
  cfr_tabular = std::shared_ptr<CFRTabular>(new CFRTabular(game));

  SPIEL_CHECK_EQ(game.NumPlayers(), 2);
  SPIEL_CHECK_EQ(game.GetType().utility, GameType::Utility::kZeroSum);
}

CFRNetModel::CFRTabular::CFRTabular(const Game& game) {}

std::vector<CFRNetModel::InferenceOutputs> CFRNetModel::CFRTabular::Inference(
    const std::vector<InferenceInputs>& inputs, bool value_or_policy,
    bool use_target) {
  std::unique_lock<std::mutex> lock(m_);
  CFRInferenceStateValueTalbe* inference_states;
  if (use_target) {
    inference_states = &target_info_states_;
  } else {
    inference_states = &info_states_;
  }
  // only one thread is allowed to edit the info_states_.
  std::vector<InferenceOutputs> outputs;
  outputs.reserve(inputs.size());
  for (auto& input : inputs) {
    uint64_t key = absl::Hash<CFRNetModel::InferenceInputs>{}(input);
    // std::cout << "inference staring ";
    auto& info_value =
        inference_states
            ->insert({key, CFRInfoStateValues(input.legal_actions, 0.000001)})
            .first->second;
    std::vector<double>* cumulative_value;
    if (value_or_policy) {
      cumulative_value = &info_value.cumulative_regrets;
    } else {
      cumulative_value = &info_value.cumulative_policy;
    }
    outputs.emplace_back(
        InferenceOutputs{input.legal_actions, *cumulative_value});
    // std::cout << "inference ending " << std::endl;
  }
  return outputs;
}

double CFRNetModel::CFRTabular::Learn(const std::vector<TrainInputs>& inputs,
                                      bool value_or_policy) {
  // only one thread is allowed to edit the info_states_.
  CFRInferenceStateValueTalbe input_states;
  std::unique_lock<std::mutex> lock(m_);
  for (auto& input : inputs) {
    InferenceInputs inf_input{input.legal_actions, input.informations};
    uint64_t key = absl::Hash<CFRNetModel::InferenceInputs>{}(inf_input);
    SPIEL_CHECK_TRUE(info_states_.find(key) != info_states_.end());
    CFRInfoStateValues& info_value = info_states_[key];
    CFRInfoStateValues& input_info_value =
        input_states
            .insert({key, CFRInfoStateValues(info_value.legal_actions, 0.0)})
            .first->second;
    std::vector<double>* cumulative_value;
    if (value_or_policy) {
      cumulative_value = &info_value.cumulative_regrets;
    } else {
      cumulative_value = &info_value.cumulative_policy;
    }
    // assume the learning rate is 0.01.
    if (value_or_policy) {
      for (int i = 0; i != input.legal_actions.size(); ++i) {
        int act_id =
            std::find(info_value.legal_actions.begin(),
                      info_value.legal_actions.end(), input.legal_actions[i]) -
            info_value.legal_actions.begin();
        SPIEL_CHECK_LT(act_id, info_value.legal_actions.size());
        input_info_value.cumulative_regrets[act_id] +=
            (input.value[i] - (*cumulative_value)[act_id]);
        input_info_value.true_regrets[act_id] += 1;
      }
    } else {
      std::vector<double> norm_policy = PolicyNormalize(*cumulative_value);
      for (int i = 0; i != input.legal_actions.size(); ++i) {
        int act_id =
            std::find(info_value.legal_actions.begin(),
                      info_value.legal_actions.end(), input.legal_actions[i]) -
            info_value.legal_actions.begin();
        double laten_value = log((*cumulative_value)[act_id]);
        input_info_value.cumulative_regrets[act_id] +=
            (input.value[i] - norm_policy[act_id]);
        input_info_value.true_regrets[act_id] += 1;
      }
    }
  }
  for (auto& input_pair : input_states) {
    SPIEL_CHECK_TRUE(info_states_.find(input_pair.first) != info_states_.end());
    CFRInfoStateValues& info_value = info_states_[input_pair.first];
    CFRInfoStateValues& input_info_value = input_pair.second;
    std::vector<double>* cumulative_value;
    if (value_or_policy) {
      cumulative_value = &info_value.cumulative_regrets;
    } else {
      cumulative_value = &info_value.cumulative_policy;
    }
    // assume the learning rate is 0.1.
    if (value_or_policy) {
      for (int i = 0; i != input_info_value.cumulative_regrets.size(); ++i) {
        (*cumulative_value)[i] += 0.01 * input_info_value.cumulative_regrets[i];
      }
      // std::cout << (*cumulative_value) << std::endl;
    } else {
      for (int i = 0; i != input_info_value.cumulative_regrets.size(); ++i) {
        (*cumulative_value)[i] =
            exp(log((*cumulative_value)[i]) +
                0.005 * input_info_value.cumulative_regrets[i]);
        ;
      }
    }
  }
  return 0.0;
}

void CFRNetModel::CFRTabular::SetValue(const TrainInputs& input,
                                       bool value_or_policy, bool accumulate,
                                       bool set_true) {
  InferenceInputs inf_input{input.legal_actions, input.informations};
  uint64_t key = absl::Hash<CFRNetModel::InferenceInputs>{}(inf_input);
  std::unique_lock<std::mutex> lock(m_);
  // std::cout << "set value staring ";
  CFRInfoStateValues& info_value =
      info_states_
          .insert({key, CFRInfoStateValues(input.legal_actions, 0.000001)})
          .first->second;
  std::vector<double>* cumulative_value;
  if (value_or_policy) {
    if (set_true) {
      cumulative_value = &info_value.true_regrets;
    } else {
      cumulative_value = &info_value.cumulative_regrets;
    }
  } else {
    cumulative_value = &info_value.cumulative_policy;
  }
  // assume the learning rate is 0.001.
  for (int i = 0; i != input.legal_actions.size(); ++i) {
    if (accumulate) {
      (*cumulative_value)[i] += input.value[i];
    } else {
      (*cumulative_value)[i] = input.value[i];
    }
  }
  // std::cout << "set value ending" << std::endl;
}

std::vector<double> CFRNetModel::CFRTabular::GetValue(
    const InferenceInputs& input, bool value_or_policy, bool get_true,
    bool normalize) {
  uint64_t key = absl::Hash<CFRNetModel::InferenceInputs>{}(input);
  std::unique_lock<std::mutex> lock(m_);
  // std::cout << "get value staring ";
  CFRInfoStateValues& info_value =
      info_states_
          .insert({key, CFRInfoStateValues(input.legal_actions, 0.000001)})
          .first->second;
  if (value_or_policy) {
    if (get_true) {
      return info_value.true_regrets;
    } else {
      return info_value.cumulative_regrets;
    }
  } else {
    if (normalize) {
      return PolicyNormalize(info_value.cumulative_policy);
    } else {
      return info_value.cumulative_policy;
    }
  }
  // std::cout << "get value ending " << std::endl;
}

std::string LoadModel(const std::string& model_path, int num_threads,
                      const std::string& device, tf::Session** tf_session,
                      tf::MetaGraphDef& meta_graph_def,
                      tf::SessionOptions& tf_opts) {
  std::cout << "loading model " << model_path << std::endl;
  TF_CHECK_OK(ReadBinaryProto(tf::Env::Default(), model_path, &meta_graph_def));

  // tf::graph::SetDefaultDevice(device, meta_graph_def.mutable_graph_def());
  // auto* graph_def = meta_graph_def.mutable_graph_def();
  // if (!device.empty()) {
  //   for (int i = 0; i < graph_def->node_size(); ++i) {
  //     graph_def->mutable_node(i)->set_device("/device:GPU:0");
  //   }
  // }
  // if (device.find("gpu") != std::string::npos) {
  //   std::string gpu_index = device.substr(std::string("/gpu:").size());
  //   tf_opts.config.mutable_gpu_options()->set_visible_device_list(gpu_index);
  // } else {
  //   tf_opts.config.mutable_gpu_options()->set_visible_device_list("");
  // }

  if ((*tf_session) != nullptr) {
    TF_CHECK_OK((*tf_session)->Close());
  }

  // create a new session
  tf_opts.config.set_intra_op_parallelism_threads(num_threads);
  tf_opts.config.set_inter_op_parallelism_threads(2);
  tf_opts.config.set_use_per_session_threads(true);
  tf_opts.config.set_allow_soft_placement(true);
  tf_opts.config.mutable_gpu_options()->set_allow_growth(true);

  TF_CHECK_OK(NewSession(tf_opts, tf_session));

  // Load graph into session
  TF_CHECK_OK((*tf_session)->Create(meta_graph_def.graph_def()));
  return model_path;
}

CFRNetModel::CFRNet::CFRNet(const Game& game, const std::string& path,
                            const std::string& model_path,
                            const std::string& file_name, int num_threads,
                            const std::string& device, bool use_target_net)
    : path_(path),
      file_name_(file_name),
      num_threads_(num_threads),
      device_(device),
      flat_input_size_(game.InformationStateTensorSize()),
      num_actions_(game.NumDistinctActions()),
      use_target_net(use_target_net) {
  if (device.find("gpu") != std::string::npos) {
    file_name_ += "_gpu";
  } else {
    file_name_ += "_cpu";
  }
  std::string load_path = absl::StrCat(model_path, "/", file_name_, ".pb");
  model_meta_graph_contents_ = file::File(load_path, "r").ReadContents();
  LoadModel(load_path, num_threads_, device, &tf_session_, meta_graph_def_,
            tf_opts_);
  // Initialize our variables
  tensorflow::RunOptions run_opt;
  // run_opt.set_inter_op_thread_pool(-1);
  TF_CHECK_OK(tf_session_->Run(run_opt, {}, {}, {"init_all_vars_op"}, nullptr,
                               nullptr));
  if (use_target_net) {
    LoadModel(load_path, num_threads_, device, &target_tf_session_,
              target_meta_graph_def_, tf_opts_);
    TF_CHECK_OK(target_tf_session_->Run(run_opt, {}, {}, {"init_all_vars_op"},
                                        nullptr, nullptr));
    SyncTarget();
  }
  auto flat = GetFlatArray();
  flat_size_ = flat.size();
}

void CFRNetModel::CFRNet::init() {
  tensorflow::RunOptions run_opt;
  // run_opt.set_inter_op_thread_pool(-1);
  tf_session_->Run(run_opt, {}, {}, {"init_all_vars_op"}, nullptr, nullptr);
}

std::string CFRNetModel::CFRNet::SaveCheckpoint(int step) {
  std::string full_path = absl::StrCat(path_, "/checkpoint-", file_name_, step);
  tf::Tensor checkpoint_path(tf::DT_STRING, tf::TensorShape());
  checkpoint_path.scalar<tf::tstring>()() = full_path;
  tensorflow::RunOptions run_opt;
  // run_opt.set_inter_op_thread_pool(-1);
  TF_CHECK_OK(tf_session_->Run(
      run_opt,
      {{meta_graph_def_.saver_def().filename_tensor_name(), checkpoint_path}},
      {}, {meta_graph_def_.saver_def().save_tensor_name()}, nullptr, nullptr));
  // Writing a checkpoint from python writes the metagraph file, but c++
  // doesn't, so do it manually to make loading checkpoints easier.
  file::File(absl::StrCat(full_path, ".meta"), "w")
      .Write(model_meta_graph_contents_);
  return full_path;
}
void CFRNetModel::CFRNet::LoadCheckpoint(const std::string& path) {
  tf::Tensor checkpoint_path(tf::DT_STRING, tf::TensorShape());
  checkpoint_path.scalar<tf::tstring>()() = path;
  tensorflow::RunOptions run_opt;
  // run_opt.set_inter_op_thread_pool(-1);
  TF_CHECK_OK(tf_session_->Run(
      run_opt,
      {{meta_graph_def_.saver_def().filename_tensor_name(), checkpoint_path}},
      {}, {meta_graph_def_.saver_def().restore_op_name()}, nullptr, nullptr));
}

void CFRNetModel::CFRNet::SetFlatArray(const Eigen::ArrayXf& flat,
                                       bool use_target) {
  tf::Session* session = tf_session_;
  if (use_target) {
    session = target_tf_session_;
  }
  tf::Tensor flat_input(tf::DT_FLOAT, tf::TensorShape({flat.size()}));
  auto flat_input_vec = flat_input.vec<float>();
  for (int i = 0; i < flat.size(); ++i) {
    flat_input_vec(i) = flat(i);
  }
  tensorflow::RunOptions run_opt;
  // run_opt.set_inter_op_thread_pool(-1);
  TF_CHECK_OK(session->Run(run_opt, {{"set_flat_input", flat_input}}, {},
                           {"set_flat"}, nullptr, nullptr));
}

Eigen::ArrayXf CFRNetModel::CFRNet::GetFlatArray(bool use_target) {
  tf::Session* session = tf_session_;
  if (use_target) {
    session = target_tf_session_;
  }
  std::vector<tf::Tensor> tf_output;
  tensorflow::RunOptions run_opt;
  // run_opt.set_inter_op_thread_pool(-1);
  TF_CHECK_OK(session->Run(run_opt, {}, {"get_flat"}, {}, &tf_output, nullptr));
  auto tf_output_vec = tf_output[0].vec<float>();
  int flat_size = tf_output_vec.size();
  // std::cout << "flat size " << flat_size << std::endl;
  Eigen::ArrayXf ret(flat_size);
  for (int i = 0; i < flat_size; ++i) {
    ret(i) = tf_output_vec(i);
  }
  return ret;
}

std::vector<CFRNetModel::InferenceOutputs> CFRNetModel::CFRNet::Inference(
    const std::vector<InferenceInputs>& inputs, bool use_target) {
  tf::Session* session = tf_session_;
  if (use_target) {
    session = target_tf_session_;
  }
  int inference_batch_size = inputs.size();
  // Fill the inputs and mask
  tf::Tensor inf_input(
      tf::DT_FLOAT, tf::TensorShape({inference_batch_size, flat_input_size_}));
  tf::Tensor inf_legals_mask(
      tf::DT_BOOL, tf::TensorShape({inference_batch_size, num_actions_}));

  auto inf_legals_mask_matrix = inf_legals_mask.matrix<bool>();
  auto inf_input_matrix = inf_input.matrix<float>();

  for (int b = 0; b < inference_batch_size; ++b) {
    // Zero initialize the sparse inputs.
    for (int a = 0; a < num_actions_; ++a) {
      inf_legals_mask_matrix(b, a) = false;
    }
    for (Action action : inputs[b].legal_actions) {
      inf_legals_mask_matrix(b, action) = true;
    }
    for (int i = 0; i < inputs[b].informations.size(); ++i) {
      inf_input_matrix(b, i) = inputs[b].informations[i];
    }
  }
  // training.set_data(std::vector<int32_t>(false));
  std::vector<tf::Tensor> inf_output;
  TF_CHECK_OK(
      session->Run({{"input", inf_input}, {"legals_mask", inf_legals_mask}},
                   {"output"}, {}, &inf_output));

  std::vector<InferenceOutputs> out;
  out.reserve(inference_batch_size);
  auto inf_output_matrix = inf_output[0].matrix<float>();
  for (int b = 0; b < inference_batch_size; ++b) {
    std::vector<double> values;
    for (Action action : inputs[b].legal_actions) {
      values.push_back(inf_output_matrix(b, action));
    }
    if (inf_output_matrix.dimension(1) > num_actions_) {
      values.push_back(
          inf_output_matrix(b, inf_output_matrix.dimension(1) - 1));
    }
    out.push_back({inputs[b].legal_actions, values});
  }

  return out;
}

double CFRNetModel::CFRNet::Learn(const std::vector<TrainInputs>& inputs,
                                  double learning_rate, double clip_norm) {
  int training_batch_size = inputs.size();
  tf::Tensor train_input(
      tf::DT_FLOAT, tf::TensorShape({training_batch_size, flat_input_size_}));
  tf::Tensor train_legals_mask(
      tf::DT_BOOL, tf::TensorShape({training_batch_size, num_actions_}));
  int num_value_size = inputs[0].value.size() == inputs[0].legal_actions.size()
                           ? num_actions_
                           : num_actions_ + 1;
  tf::Tensor targets(tf::DT_FLOAT,
                     tf::TensorShape({training_batch_size, num_value_size}));
  tf::Tensor weights(tf::DT_FLOAT, tf::TensorShape({training_batch_size, 1}));
  tf::Tensor gradient_input(tf::DT_FLOAT, tf::TensorShape({flat_size_}));
  tf::Tensor learning_rate_input(tf::DT_FLOAT, tf::TensorShape({}));
  tf::Tensor clip_norm_input(tf::DT_FLOAT, tf::TensorShape({}));

  auto train_input_matrix = train_input.matrix<float>();
  auto train_legals_mask_matrix = train_legals_mask.matrix<bool>();
  auto targets_matrix = targets.matrix<float>();
  auto weights_matrix = weights.matrix<float>();
  auto gradient_input_matrix = gradient_input.vec<float>();
  auto learning_rate_scalar = learning_rate_input.scalar<float>();
  auto clip_norm_scalar = clip_norm_input.scalar<float>();
  double loss = 0;

  learning_rate_scalar(0) = learning_rate;
  clip_norm_scalar(0) = clip_norm;

  if (!training_batch_size) {
    loss = 0;
  } else {
    // #pragma omp parallel for default(shared) schedule(dynamic)
    for (int b = 0; b < training_batch_size; ++b) {
      // Zero initialize the sparse inputs.
      for (int a = 0; a < num_actions_; ++a) {
        train_legals_mask_matrix(b, a) = false;
        targets_matrix(b, a) = 0;
      }
      for (Action action : inputs[b].legal_actions) {
        train_legals_mask_matrix(b, action) = true;
      }
      for (int i = 0; i < inputs[b].informations.size(); ++i) {
        train_input_matrix(b, i) = inputs[b].informations[i];
      }
      for (int i = 0; i != inputs[b].legal_actions.size(); ++i) {
        targets_matrix(b, inputs[b].legal_actions[i]) = inputs[b].value[i];
      }
      if (num_value_size == num_actions_ + 1) {
        targets_matrix(b, num_actions_) = inputs[b].value.back();
      }
      weights_matrix(b, 0) = inputs[b].weight;
    }
    std::vector<tf::Tensor> tf_outputs;
    tensorflow::RunOptions run_opt;
    // run_opt.set_inter_op_thread_pool(-1);
    // run_opt.set_report_tensor_allocations_upon_oom(true);
    // tensorflow::RunMetadata run_meta;
    TF_CHECK_OK(tf_session_->Run(run_opt,
                                 {{"input", train_input},
                                  {"legals_mask", train_legals_mask},
                                  {"targets", targets},
                                  {"weights", weights},
                                  {"learning_rate", learning_rate_input},
                                  {"clip_norm", clip_norm_input}},
                                 {"loss"}, {"sim_train"}, &tf_outputs,
                                 nullptr));
    loss = tf_outputs[0].scalar<float>()(0);
  }
  return loss;
}

}  // namespace algorithms
}  // namespace open_spiel
