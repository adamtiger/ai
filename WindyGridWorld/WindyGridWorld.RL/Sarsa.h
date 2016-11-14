#ifndef __SARSA__
#define __SARSA__

#include "IAlgorithm.h"

namespace native {
	namespace alg {

		class Sarsa : public frw::IAlgorithm {

		public:

			Sarsa(double alpha, double gamma);

			virtual void DoOneLearningIterate();

		};

	} // alg
} // native

#endif // __SARSA__
