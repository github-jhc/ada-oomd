
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

#ifndef THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_OUTCOME_SAMPLING_MCCFR_H_
#define THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_OUTCOME_SAMPLING_MCCFR_H_

#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

#include "deep_escfr_solver.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/public_tree.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

class DeepOSCFRSolver : public DeepESCFRSolver {
 public:
  DeepOSCFRSolver(const Game& game,
                  std::shared_ptr<VPNetEvaluator> value_0_eval_,
                  std::shared_ptr<VPNetEvaluator> value_1_eval_,
                  std::shared_ptr<VPNetEvaluator> policy_eval,
                  bool use_regret_net, bool use_policy_net, bool use_tabular,
                  std::mt19937* rng,
                  AverageType avg_type = AverageType::kSimple)
      : DeepESCFRSolver(game, value_0_eval_, value_1_eval_, policy_eval,
                        use_regret_net, use_policy_net, use_tabular, rng,
                        avg_type) {}

 protected:
  double UpdateRegrets(PublicNode* node, Player player, double player_reach,
                       double oppoment_reach, double sampling_reach,
                       Trajectory& value_trajectory,
                       Trajectory& policy_trajectory, int step,
                       std::mt19937* rng,
                       const ChanceData& chance_data) override;
};

}  // namespace algorithms
}  // namespace open_spiel
#endif  // THIRD_PARTY_OPEN_SPIEL_ALGORITHMS_OUTCOME_SAMPLING_MCCFR_H_