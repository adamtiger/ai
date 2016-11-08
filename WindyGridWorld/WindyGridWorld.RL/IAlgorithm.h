#ifndef __I_ALGORITHM__
#define __I_ALGORITHM__

#include "Environment.h"
#include "ActionSelectors.h"

namespace native {
	namespace frw {

		class IAlgorithm {

		public:
			virtual void DoOneLearningIterate() = 0;
			virtual void SetEnvironment(Environment* env) = 0;
		};
	}
}

#endif // __I_ALGORITHM__
