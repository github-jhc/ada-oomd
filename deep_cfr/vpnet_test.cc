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

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tf = tensorflow;
using Tensor = Eigen::Tensor<float, 2, Eigen::RowMajor>;
using TensorMap = Eigen::TensorMap<Tensor, Eigen::Aligned>;
using TensorBool = Eigen::Tensor<bool, 2, Eigen::RowMajor>;
using TensorMapBool = Eigen::TensorMap<TensorBool, Eigen::Aligned>;

namespace open_spiel {
namespace algorithms {

namespace {

CFRNetModel BuildModel(const Game& game, const std::string& nn_model,
                       bool create_graph, bool use_target = false) {
  std::string tmp_dir = open_spiel::file::GetTmpDir();
  std::string filename = absl::StrCat("vp_net_test_", nn_model);

  if (create_graph) {
    CreateModel(game,
                /*learning_rate=*/0.001,
                /*weight_decay=*/0.0, tmp_dir, filename, nn_model, nn_model,
                /*nn_width=*/64, /*nn_depth=*/1, false);
  }

  CFRNetModel model(game, tmp_dir, tmp_dir, filename, 1, "/cpu:0", true, false,
                    false, false, use_target);

  return model;
}

void TestTensorflow() {
  tf::Session* session;
  TF_CHECK_OK(NewSession(tf::SessionOptions(), &session));
  std::cout << "Session successfully created." << std::endl;
}

GameParameters LeducPokerParameters() {
  return {{"betting", GameParameter(std::string("limit"))},
          {"numPlayers", GameParameter(2)},
          {"numRounds", GameParameter(2)},
          {"blind", GameParameter(std::string("1 1"))},
          {"raiseSize", GameParameter(std::string("2 4"))},
          {"firstPlayer", GameParameter(std::string("1 1"))},
          {"maxRaises", GameParameter(std::string("2 2"))},
          {"numSuits", GameParameter(2)},
          {"numRanks", GameParameter(3)},
          {"numHoleCards", GameParameter(1)},
          {"numBoardCards", GameParameter(std::string("0 1"))}};
}

void TestModelCreation(const std::string& nn_model) {
  std::cout << "TestModelCreation: " << nn_model << std::endl;
  std::shared_ptr<const Game> game =
      LoadGame("universal_poker", LeducPokerParameters());
  CFRNetModel model = BuildModel(*game, nn_model, true);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    std::vector<Action> legal_actions = state->LegalActions();
    Action action = legal_actions[0];
    state->ApplyAction(action);
  }
  std::vector<Action> legal_actions = state->LegalActions();
  std::vector<double> obs = state->InformationStateTensor();
  CFRNetModel::InferenceInputs inputs = {legal_actions, obs};

  // Check that inference runs at all.
  auto out = model.InfValue(0, {inputs});
  std::cout << out << std::endl;

  std::vector<CFRNetModel::TrainInputs> train_inputs;
  train_inputs.emplace_back(CFRNetModel::TrainInputs{
      legal_actions, obs, std::vector<double>(legal_actions.size(), 0.1), 1});

  // Check that learning runs at all.
  auto loss = model.TrainValue(0, train_inputs, 0.001);
  std::cout << "loss " << loss << std::endl;
}

// Can learn a single trajectory
void TestModelLearnsSimple(const std::string& nn_model) {
  std::cout << "TestModelLearnsSimple: " << nn_model << std::endl;
  std::shared_ptr<const Game> game =
      LoadGame("universal_poker", LeducPokerParameters());
  CFRNetModel model = BuildModel(*game, nn_model, false);

  std::vector<CFRNetModel::TrainInputs> train_inputs;
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> dist;
  while (!state->IsTerminal()) {
    std::vector<Action> legal_actions = state->LegalActions();
    int action_ind = dist(rng, std::uniform_int_distribution<int>::param_type{
                                   0, legal_actions.size() - 1});
    Action action = legal_actions[action_ind];
    if (!state->IsChanceNode()) {
      std::vector<double> obs = state->InformationStateTensor();
      std::vector<double> target;
      if (nn_model == "normal") {
        for (int i = 0; i != legal_actions.size(); ++i) {
          target.push_back(
              dist(rng, std::uniform_int_distribution<int>::param_type{0, 9}));
        }
      } else if (nn_model == "softmax") {
        for (int i = 0; i != legal_actions.size(); ++i) {
          target.push_back(0);
        }
        target[action_ind] = 1;
      }
      train_inputs.emplace_back(
          CFRNetModel::TrainInputs{legal_actions, obs, target, 1});

      CFRNetModel::InferenceInputs inputs = {legal_actions, obs};
      std::vector<CFRNetModel::InferenceOutputs> out =
          model.InfValue(0, {inputs});
      SPIEL_CHECK_EQ(out.size(), 1);
      SPIEL_CHECK_EQ(out[0].value.size(), legal_actions.size());
    }

    state->ApplyAction(action);
  }

  std::cout << "states: " << train_inputs.size() << std::endl;
  for (auto& ti : train_inputs) {
    std::cout << ti << std::endl;
  }
  std::vector<double> losses;
  for (int i = 0; i < 1000; i++) {
    double loss = model.TrainValue(0, train_inputs, 0.001);
    std::cout << absl::StrFormat("loss: %.3f", loss) << std::endl;
    losses.push_back(loss);
    if (loss < 0.05 && i > 2) {
      break;
    }
  }
  SPIEL_CHECK_GT(losses.front(), losses.back());
  SPIEL_CHECK_LT(losses.back(), 0.05);
  SPIEL_CHECK_LT(losses.back(), 0.05);
}

// Can learn a single trajectory
void TestModelSaveAndRestore(const std::string& nn_model) {
  std::cout << "TestModelSaveAndRestore: " << nn_model << std::endl;
  std::shared_ptr<const Game> game =
      LoadGame("universal_poker", LeducPokerParameters());
  CFRNetModel model = BuildModel(*game, nn_model, false);

  std::vector<CFRNetModel::TrainInputs> train_inputs;
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    std::vector<Action> legal_actions = state->LegalActions();
    Action action = legal_actions[0];
    state->ApplyAction(action);
  }
  std::vector<Action> legal_actions = state->LegalActions();
  std::vector<double> obs = state->InformationStateTensor();
  CFRNetModel::InferenceInputs inputs = {legal_actions, obs};

  // Check that inference runs at all.
  std::cout << "after loading:" << std::endl;
  auto out = model.InfValue(0, {inputs});
  std::cout << out << std::endl;
  auto path = model.SaveValue(0, 0);
  model.InitValue(0);
  std::cout << "after reinit:" << std::endl;
  auto out1 = model.InfValue(0, {inputs});
  std::cout << out1 << std::endl;
  model.RestoreValue(0, path);
  std::cout << "after restore:" << std::endl;
  auto out2 = model.InfValue(0, {inputs});
  std::cout << out2 << std::endl;
}

void TestSyncFrom(const std::string& nn_model) {
  std::cout << "TestSyncFrom: " << nn_model << std::endl;
  std::shared_ptr<const Game> game =
      LoadGame("universal_poker", LeducPokerParameters());
  CFRNetModel model = BuildModel(*game, nn_model, false);
  CFRNetModel model_2 = BuildModel(*game, nn_model, false);

  std::vector<CFRNetModel::TrainInputs> train_inputs;
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    std::vector<Action> legal_actions = state->LegalActions();
    Action action = legal_actions[0];
    state->ApplyAction(action);
  }
  std::vector<Action> legal_actions = state->LegalActions();
  std::vector<double> obs = state->InformationStateTensor();
  CFRNetModel::InferenceInputs inputs = {legal_actions, obs};

  // Check that inference runs at all.
  std::cout << "model 1:" << std::endl;
  auto out = model.InfValue(0, {inputs});
  std::cout << out << std::endl;
  std::cout << "model 2:" << std::endl;
  auto outm2 = model_2.InfValue(0, {inputs});
  std::cout << outm2 << std::endl;
  model.InitValue(0);
  std::cout << "after reinit, model 1:" << std::endl;
  auto out1 = model.InfValue(0, {inputs});
  std::cout << out1 << std::endl;
  model.SyncValueFrom(0, model_2);
  std::cout << "after sync from model 2, model 1:" << std::endl;
  auto out2 = model.InfValue(0, {inputs});
  std::cout << out2 << std::endl;
  model_2.InitValue(0);
  std::cout << "after reinit, model 2:" << std::endl;
  auto out1m2 = model_2.InfValue(0, {inputs});
  std::cout << out1m2 << std::endl;
  model.SyncValueFrom(0, model_2);
  std::cout << "after sync from model 2, model 1:" << std::endl;
  auto out3 = model.InfValue(0, {inputs});
  std::cout << out3 << std::endl;
}

void TestSyncTarget(const std::string& nn_model) {
  std::cout << "TestSyncTarget: " << nn_model << std::endl;
  std::shared_ptr<const Game> game =
      LoadGame("universal_poker", LeducPokerParameters());
  CFRNetModel model = BuildModel(*game, nn_model, false, true);

  std::vector<CFRNetModel::TrainInputs> train_inputs;
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    std::vector<Action> legal_actions = state->LegalActions();
    Action action = legal_actions[0];
    state->ApplyAction(action);
  }
  std::vector<Action> legal_actions = state->LegalActions();
  std::vector<double> obs = state->InformationStateTensor();
  CFRNetModel::InferenceInputs inputs = {legal_actions, obs};

  // Check that inference runs at all.
  std::cout << "model:" << std::endl;
  auto out = model.InfValue(0, {inputs});
  std::cout << out << std::endl;
  std::cout << "model target:" << std::endl;
  auto outm2 = model.InfTargetValue(0, {inputs});
  std::cout << outm2 << std::endl;
  model.InitValue(0);
  std::cout << "after reinit, model:" << std::endl;
  auto out1 = model.InfValue(0, {inputs});
  std::cout << out1 << std::endl;
  model.SyncValue(0);
  std::cout << "after sync target, model target:" << std::endl;
  auto out2 = model.InfTargetValue(0, {inputs});
  std::cout << out2 << std::endl;
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::TestModelCreation("normal");
  // open_spiel::algorithms::TestModelCreation("baseline");
  // open_spiel::algorithms::TestModelCreation("softmax");

  // Tests below here reuse the graphs created above. Graph creation is slow
  // due to calling a separate python process.

  open_spiel::algorithms::TestModelLearnsSimple("normal");
  // open_spiel::algorithms::TestModelLearnsSimple("baseline");
  // open_spiel::algorithms::TestModelLearnsSimple("softmax");

  open_spiel::algorithms::TestModelSaveAndRestore("normal");
  open_spiel::algorithms::TestSyncFrom("normal");
  open_spiel::algorithms::TestSyncTarget("normal");
}
