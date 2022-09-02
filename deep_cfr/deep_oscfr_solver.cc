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

#include "deep_oscfr_solver.h"

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
double DeepOSCFRSolver::UpdateRegrets(
    PublicNode* node, Player player, double player_reach, double opponent_reach,
    double sampling_reach, Trajectory& value_trajectory,
    Trajectory& policy_trajectory, int step, std::mt19937* rng,
    const ChanceData& chance_data) {
  State& state = *(node->GetState());
  state.SetChance(chance_data);
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
    double new_sampling_reach =
        info_state_copy.current_policy[aidx] * sampling_reach;
    value =
        UpdateRegrets(node->GetChild(legal_actions[aidx]), player, player_reach,
                      new_reach, new_sampling_reach, value_trajectory,
                      policy_trajectory, step, rng, chance_data);
  } else {
    double epsilon = 0.6;
    int aidx = info_state_copy.SampleActionIndex(epsilon, dist_(*rng));
    double new_sampling_reach =
        ((1 - epsilon) * info_state_copy.current_policy[aidx] +
         epsilon / legal_actions.size()) *
        sampling_reach;
    double child_reach = info_state_copy.current_policy[aidx] * player_reach;
    child_values[aidx] =
        UpdateRegrets(node->GetChild(legal_actions[aidx]), player, child_reach,
                      opponent_reach, new_sampling_reach, value_trajectory,
                      policy_trajectory, step, rng, chance_data);
    child_values[aidx] /=
        ((1 - epsilon) * info_state_copy.current_policy[aidx] +
         epsilon / legal_actions.size());
    value += info_state_copy.current_policy[aidx] * child_values[aidx];
  }

  if (cur_player == player) {
    // NOTE: only for debug.
    if (!use_regret_net || use_tabular) {
      std::vector<double> linear_regret(legal_actions.size());
      for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
        linear_regret[aidx] = (child_values[aidx] - value) * step *
                              opponent_reach / sampling_reach;
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
      value_trajectory.states.push_back(ReplayNode{
          information_tensor, cur_player, legal_actions, regret, player_reach,
          (double)step * opponent_reach / sampling_reach});
    }
  }
  if (avg_type_ == AverageType::kSimple && cur_player != player) {
    if (!use_policy_net || use_tabular) {
      std::vector<double> linear_policy(legal_actions.size());
      for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
        linear_policy[aidx] = info_state_copy.current_policy[aidx] *
                              opponent_reach / sampling_reach * step;
      }
      CFRNetModel::TrainInputs train_input{legal_actions, information_tensor,
                                           linear_policy};
      policy_eval_->AccumulateCFRTabular(train_input);
    }
    if (use_policy_net) {
      std::vector<double> policy(legal_actions.size());
      for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
        policy[aidx] = info_state_copy.current_policy[aidx];
      }
      policy_trajectory.states.push_back(ReplayNode{
          information_tensor, cur_player, legal_actions, policy, opponent_reach,
          (double)step * opponent_reach / sampling_reach});
    }
  }
  return value;
}

}  // namespace algorithms
}  // namespace open_spiel
