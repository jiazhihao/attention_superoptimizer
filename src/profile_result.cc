#include "aso/profile_result.h"

#include <limits>

namespace aso {

ProfileResult ProfileResult::infinity() {
  return ProfileResult{std::numeric_limits<float>::infinity()};
}

}