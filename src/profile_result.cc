#include "aso/profile_result.h"

#include <limits>

namespace aso {

ProfileResult ProfileResult::infinity() {
  return ProfileResult{1000};
}

}