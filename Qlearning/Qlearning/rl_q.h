#ifndef __RL_Q__
#define __RL_Q__

#include "rl_q_utils.h"

namespace rl {
	namespace q {

		class Q {

		public:

			explicit Q() {}
			Q(float alpha, float gamma) {
				_alpha = alpha;
				_gamma = gamma;
				init_ac_st_vals_randomly();
			}

			void UpdateActionStateValue(float reward, tAction act, tState current, tState next) {

				_acv(current, act) = _acv(current, act) + _alpha * (reward + _gamma * max(next) - _acv(current, act));
			}
	
		private:

			ac_st_value _acv;
			float _gamma;
			float _alpha;


			// Functions.

			void init_ac_st_vals_randomly() {
				_acv.InitRand();
			}

			float& max(tState next) const{


			}
		};
	}
}
#endif // __RL_Q__
