#ifndef __RL_Q__
#define __RL_Q__

#include "rl_q_utils.h"

namespace rl {
	namespace q {

		class Q {

		public:

			explicit Q(int numst):
				_acv(numst), policy(_acv){}
			Q(float alpha, float gamma, int numst):
				_acv(numst), policy(_acv)
			{
				_alpha = alpha;
				_gamma = gamma;
				_numst = numst;
			}

			void UpdateActionStateValue(float reward, tAction act, tState current, tState next) {

				_acv(current, act) = _acv(current, act) + _alpha * (reward + _gamma * maxx(next) - _acv(current, act));
			}

			tAction ObserveAction(tState current) {
				return policy.use_policy(current);
			}

			void DecayAlpha(float ratio = 1.0f) {
				_alpha = _alpha / ratio;
			}
	
		private:

			ac_st_value _acv;
			Policy policy;
			float _gamma;
			float _alpha;
			int _numst;


			// Functions.

			float& maxx(tState next) const{

				float max_val = 0.0f;
				for (auto it : _acv(next)) {
					if (it > max_val) max_val = it;
				}

				return max_val;
			}
		};
	}
}
#endif // __RL_Q__
