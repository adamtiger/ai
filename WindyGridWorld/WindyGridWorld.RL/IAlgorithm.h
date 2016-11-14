#ifndef __I_ALGORITHM__
#define __I_ALGORITHM__

#include "Environment.h"
#include "ActionSelectors.h"

namespace native {
	namespace frw {

		class IAlgorithm {

		public:

			IAlgorithm(double alpha, double gamma) :
				alpha_(alpha), gamma_(gamma) {}

			virtual void DoOneLearningIterate() = 0;

			virtual void SetEnvironment(native::frw::Environment * env) {
				env_ = env;

				qsa_.resize(env_->GetValuesNumber());

				for (int i = 0; i < env_->GetValuesNumber(); ++i) {
					qsa_[i].resize(4); // four actions are possible
				}

				for (int i = 0; i < env_->GetValuesNumber(); ++i) {
					for (int j = 0; j < 4; ++j) {
						qsa_[i][j] = 0.0;
					}
				}
			}

		protected:

			virtual int max(int idx) const {

				int place_max = 0;

				for (int i = 1; i < 4; ++i) {
					if (qsa_[idx][i] > qsa_[idx][place_max])
						place_max = i;
				}

				return place_max;
			}

			// Variables
			frw::Environment* env_;

			double alpha_;
			double gamma_;

			vector<vector<double>> qsa_;
		};
	}
}

#endif // __I_ALGORITHM__
