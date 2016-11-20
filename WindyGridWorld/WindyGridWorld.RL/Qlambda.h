#ifndef __Q_LAMBDA__
#define __Q_LAMBDA__

#include "IAlgorithm.h"

namespace native {
	namespace alg {

		class QLambda : public frw::IAlgorithm {

		public:

			QLambda(double alpha, double gamma, double lambda = 0.5);

			virtual void DoOneLearningIterate();

			virtual void SetEnvironment(native::frw::Environment * env);

		private:

			vector<vector<double>> zsa_;

			double lambda_;

		};
	} // alg
} // native

#endif // __Q_LAMBDA__
