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

#include "deep_escfr_solver.h"

#include <memory>
#include <numeric>
#include <random>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

DeepESCFRSolver::DeepESCFRSolver(const Game& game,
                                 std::shared_ptr<VPNetEvaluator> value_0_eval,
                                 std::shared_ptr<VPNetEvaluator> value_1_eval,
                                 std::shared_ptr<VPNetEvaluator> policy_eval,
                                 bool use_regret_net, bool use_policy_net,
                                 bool use_tabular, std::mt19937* rng,
                                 AverageType avg_type)
    : game_(game.Clone()),
      rng_(rng),
      iterations_(0),
      avg_type_(avg_type),
      dist_(0.0, 1.0),
      value_eval_{value_0_eval, value_1_eval},
      policy_eval_(policy_eval),
      tree_(game_->NewInitialState()),
      root_node_(tree_.Root()),
      root_state_(root_node_->GetState()),
      use_regret_net(use_regret_net),
      use_policy_net(use_policy_net),
      use_tabular(use_tabular) {
  if (game_->GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError(
        "MCCFR requires sequential games. If you're trying to run it "
        "on a simultaneous (or normal-form) game, please first transform it "
        "using turn_based_simultaneous_game.");
  }
}
std::pair<std::vector<Trajectory>, std::vector<Trajectory>>
DeepESCFRSolver::RunIteration() {
  std::vector<Trajectory> value_trajectories(game_->NumPlayers());
  std::vector<Trajectory> policy_trajectories(game_->NumPlayers());
  for (auto p = Player{0}; p < game_->NumPlayers(); ++p) {
    auto ret_p = RunIteration(rng_, p, 0);
    value_trajectories.push_back(ret_p.first);
    policy_trajectories.push_back(ret_p.second);
  }
  return {value_trajectories, policy_trajectories};
}

std::pair<Trajectory, Trajectory> DeepESCFRSolver::RunIteration(Player player,
                                                                int step) {
  return RunIteration(rng_, player, step);
}

std::pair<Trajectory, Trajectory> DeepESCFRSolver::RunIteration(
    std::mt19937* rng, Player player, int step) {
  ++iterations_;
  node_touch_ = 0;
  Trajectory value_trajectory;
  Trajectory policy_trajectory;
  // Sample a chace seed at the start of an iteration.
  ChanceData chance_data = root_state_->SampleChance(rng);
  // NOTE: We do not need to clearCache if the networks are never updated. So
  // the Cache should be clear by the learner. Don't do this:
  // value_eval_->ClearCache();
  UpdateRegrets(root_node_, player, 1, 1, 1, value_trajectory,
                policy_trajectory, step, rng, chance_data);

  return {value_trajectory, policy_trajectory};
}

double DeepESCFRSolver::UpdateRegrets(
    PublicNode* node, Player player, double player_reach, double opponent_reach,
    double sampling_reach, Trajectory& value_trajectory,
    Trajectory& policy_trajectory, int step, std::mt19937* rng,
    const ChanceData& chance_data) {
  State& state = *(node->GetState());
  state.SetChance(chance_data);
  // std::cout << state.ToString() << std::endl;
  if (state.IsTerminal()) {
    return state.PlayerReturn(player);
  } else if (state.IsChanceNode()) {
    Action action = SampleAction(state.ChanceOutcomes(), dist_(*rng)).first;
    return UpdateRegrets(node->GetChild(action), player, player_reach,
                         opponent_reach, sampling_reach, value_trajectory,
                         policy_trajectory, step, rng, chance_data);
  } else if (state.IsSimultaneousNode()) {
    SpielFatalError(
        "Simultaneous moves not supported. Use "
        "TurnBasedSimultaneousGame to convert the game first.");
  }

  node_touch_ += 1;
  Player cur_player = state.CurrentPlayer();
  std::string is_key = state.InformationStateString(cur_player);
  std::vector<Action> legal_actions = state.LegalActions();
  std::vector<double> information_tensor = state.InformationStateTensor();

  // NOTE: why we need a copy here? don't copy, just create one.
  CFRInfoStateValues info_state_copy(legal_actions, kInitialTableValues);
  if (step != 1) {
    auto cfr_value = value_eval_[cur_player]->Inference(state);
    std::vector<double> cfr_regret = cfr_value.value;
    info_state_copy.SetRegret(cfr_regret);
  }
  info_state_copy.ApplyRegretMatchingUsingMax();

  double value = 0.0;
  std::vector<double> child_values(legal_actions.size(), 0.0);

  if (cur_player != player) {
    // Sample at opponent nodes.
    int aidx = info_state_copy.SampleActionIndex(0.0, dist_(*rng));
    double new_reach = info_state_copy.current_policy[aidx] * opponent_reach;
    value =
        UpdateRegrets(node->GetChild(legal_actions[aidx]), player, player_reach,
                      new_reach, sampling_reach, value_trajectory,
                      policy_trajectory, step, rng, chance_data);
  } else {
    // Walk over all actions at my nodes
    for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
      double child_reach = info_state_copy.current_policy[aidx] * player_reach;
      child_values[aidx] = UpdateRegrets(
          node->GetChild(legal_actions[aidx]), player, child_reach,
          opponent_reach, sampling_reach, value_trajectory, policy_trajectory,
          step, rng, chance_data);
      value += info_state_copy.current_policy[aidx] * child_values[aidx];
    }
  }

  if (cur_player == player) {
    // NOTE: only for debug.
    if (!use_regret_net || use_tabular) {
      std::vector<double> linear_regret(legal_actions.size());
      for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
        linear_regret[aidx] = (child_values[aidx] - value) * step;
      }
      CFRNetModel::TrainInputs train_input{legal_actions, information_tensor,
                                           linear_regret};
      value_eval_[cur_player]->AccumulateCFRTabular(train_input);
    }
    if (use_regret_net) {
      std::vector<double> regret(legal_actions.size());
      for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
        regret[aidx] = (child_values[aidx] - value);
      }
      value_trajectory.states.push_back(
          ReplayNode{information_tensor, cur_player, legal_actions, regret,
                     player_reach, (double)step});
    }
  }

  // Simple average does averaging on the opponent node. To do this in a game
  // with more than two players, we only update the player + 1 mod num_players,
  // which reduces to the standard rule in 2 players.
  if (avg_type_ == AverageType::kSimple &&
      cur_player == ((player + 1) % game_->NumPlayers())) {
    // NOTE: only for debug.
    if (!use_policy_net || use_tabular) {
      std::vector<double> linear_policy(legal_actions.size());
      for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
        linear_policy[aidx] = info_state_copy.current_policy[aidx] * step;
      }
      CFRNetModel::TrainInputs train_input{legal_actions, information_tensor,
                                           linear_policy};
      policy_eval_->AccumulateCFRTabular(train_input);
    }
    if (use_policy_net) {
      policy_trajectory.states.push_back(ReplayNode{
          information_tensor, cur_player, legal_actions,
          info_state_copy.current_policy, opponent_reach, (double)step});
    }
  }
  return value;
}

}  // namespace algorithms
}  // namespace open_spiel
