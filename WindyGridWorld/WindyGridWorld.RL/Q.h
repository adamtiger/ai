#ifndef __Q__
#define __Q__

#include "IAlgorithm.h"

namespace native {
	namespace alg {

		class Q : public frw::IAlgorithm {
		public:

			Q(double alpha, double gamma);

			virtual void DoOneLearningIterate();
			virtual void SetEnvironment(frw::Environment* env);
		};
	}
}

#endif // __Q__
