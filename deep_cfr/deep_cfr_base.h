#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "deep_cfr.h"
#include "device_manager.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/games/universal_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/data_logger.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/utils/logger.h"
#include "open_spiel/utils/reservior_buffer.h"
#include "open_spiel/utils/stats.h"
#include "open_spiel/utils/thread.h"
#include "universal_poker_exploitability.h"
#include "vpevaluator.h"
#include "vpnet.h"

std::string SaveAndLoad(const DeepCFRConfig& config,
                        DeviceManager* device_manager, int step,
                        bool value_or_policy, Player player, int device_id,
                        bool save_policy = false) {
  // Always save a checkpoint, either for keeping or for loading the weights
  // to the other sessions. It only allows numbers, so use -1 as "latest".
  std::string checkpoint_path;
  if (value_or_policy) {
    if (step % config.checkpoint_freq == 0 || config.sync_by_restore) {
      checkpoint_path =
          device_manager->Get(0, device_id)
              ->SaveValue(player,
                          step % config.checkpoint_freq == 0 ? step : -1);
      if (device_manager->Count() > 0 && config.sync_by_restore) {
        for (auto& device : device_manager->GetAll()) {
          device->RestoreValue(player, checkpoint_path);
        }
      }
    }
  } else {
    if (step % config.checkpoint_freq == 0 || config.sync_by_restore ||
        save_policy) {
      checkpoint_path =
          device_manager->Get(0, device_id)
              ->SavePolicy(player,
                           step % config.checkpoint_freq == 0 ? step : -1);
      if (device_manager->Count() > 0 && config.sync_by_restore) {
        for (auto& device : device_manager->GetAll()) {
          device->RestorePolicy(player, checkpoint_path);
        }
      }
    }
  }
  return checkpoint_path;
}

void SyncModels(const DeepCFRConfig& config, DeviceManager* device_manager,
                int step, bool value_or_policy, Player player, int device_id) {
  if (value_or_policy) {
    for (auto& device : device_manager->GetAll()) {
      device->SyncValueFrom(player,
                            *(device_manager->Get(0, device_id).model()));
    }
  } else {
    for (auto& device : device_manager->GetAll()) {
      device->SyncPolicyFrom(player,
                             *(device_manager->Get(0, device_id).model()));
    }
  }
}

std::unordered_set<std::string> GetEvalInfos(const DeepCFRConfig& config) {
  std::unordered_set<std::string> infos;
  if (config.game == "leduc_poker") {
    infos = {
        "Qh:Qs:crc,r",   "As:Qh:crrc,r",  "Kh:Ah:rc,r",   "Qh:Qs:crc,cr",
        "Kh:Qh:crc,c",   "Qs:As:rrc,cr",  "Ks:Qh:cc,cr",  "As:Qs:crrc",
        "Qh:Kh:crrc,cr", "Qh::crr",       "Qs:Ks:rc,crr", "Ah:Kh:crrc,rr",
        "Ks:As:crc,crr", "Ks:Qh:rrc,crr", "Kh:Ks:cc,rr",  "Qs:Kh:cc,r",
        "As::",          "Ks:As:crc,rr",  "Kh:Qs:cc,crr", "As:Qs:rc,cr",
        "Ks::r",         "As:Ah:crrc,c",  "Qh::rr",       "Ah:Ks:rrc,r",
    };
  } else if (config.game.find("FHP") != std::string::npos) {
    infos = {
        "QhAs:JsKhKs:crrc,crrrr",
        "KsAh:ThJhJs:crc,crr",
        "QhAh:TsJsQs:crc,rrrr",
        "QsKh:ThKsAh:crrrc,rrrr",
        "TsKs:ThAhAs:rc,crrrr",
        "QsKh:TsJsAs:crrc,cr",
        "JhKh:KsAhAs:crc,crrrr",
        "ThKs::cr",
        "ThAs:TsJsQs:rc",
        "TsAs:ThJhKh:rrc,rr",
        "ThKh:JsQsAh:cc,r",
        "JhJs:ThKsAh:crrc,rr",
        "TsAh:QhQsKs:rrc,r",
        "QhKh::crrr",
        "QsAh:JsQhKs:rc,rrr",
        "KsAh:ThKhAs:cc,rr",
        "QhKh::crr",
        "KsAs:TsQhKh:cc,crrrr",
        "JhQh:ThTsAs:rrrc,rrr",
        "QsAh::rrr",
        "JsAs:QhQsKh:crrrc,crrrr",
        "QhKs:ThQsAs:rc,r",
        "TsAh:JhJsKs:crrrc,c",
        "JhAh:QsKhAs:crc,rrr",
        "JsKh:QhAhAs:rc,crr",
        "KhAs:JhJsKs:rrrc,rrrr",
        "QhAs::c",
        "TsJh:QhQsAs:rrrc,r",
        "TsQs:ThJsKh:crrrc,crr",
        "AhAs:JhQhKh:cc,rrr",
        "QsAh:JsQhAs:crrrc,rr",
        "JhKh:ThJsQs:crrc,rrr",
        "TsKs:JhJsQh:rrrc,c",
        "ThKs:TsQhAh:rrrc,cr",
        "QsAs:JhQhAh:cc,rrrr",
        "ThQs:QhKhKs:crrrc",
        "TsQs:ThJhAh:rrc,crrr",
        "QsKh:JhJsKs:rc,rrrr",
        "AhAs:TsJsKh:crrc",
        "JsAh:TsJhKs:rrrc",
        "AhAs:JhQhQs:rc,crrr",
        "AhAs:QhKhKs:cc",
        "ThKs:TsKhAh:rrrc,crrrr",
        "KsAh::",
        "TsKs:JsKhAh:crc,r",
        "QhQs:TsJhJs:cc,c",
        "QhAs:ThJhKs:crrc,c",
        "QhAs:JsQsAh:crc,crrr",
        "ThQs::r",
        "ThKs:JhQhKh:rc,cr",
    };
  } else {
    SPIEL_CHECK_TRUE(config.game == "kuhn_poker");
  }
}
