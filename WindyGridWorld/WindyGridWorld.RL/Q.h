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

		private:

			int max(int idx) const;

			frw::Environment* env_;

			double alpha_;
			double gamma_;

			vector<vector<double>> qsa_;
		};
	}
}

#endif // __Q__
