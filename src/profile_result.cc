#include "mirage/profile_result.h"

#include <limits>

namespace mirage {

ProfileResult ProfileResult::infinity() {
  return ProfileResult{std::numeric_limits<float>::infinity()};
}

}
