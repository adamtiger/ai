#ifndef __SARSA_LAMBDA__
#define __SARSA_LAMBDA__

#include "IAlgorithm.h"

namespace native {
	namespace alg {

		class SarsaLambda : public frw::IAlgorithm {

		public:

			SarsaLambda(double alpha, double gamma, double lambda = 0.5);

			virtual void DoOneLearningIterate();

			virtual void SetEnvironment(native::frw::Environment * env);

		private:
			
			vector<vector<double>> zsa_;

			double lambda_;

		};
	} // alg
} // native

#endif // __SARSA_LAMBDA__
