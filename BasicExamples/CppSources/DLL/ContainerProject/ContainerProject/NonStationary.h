#ifndef __NON_STATIONARY__
#define __NON_STATIONARY__

#include "Environment.h"

namespace bandit {
	namespace nonstationary {

		class NonStationary : public BaseEnvironment {

		public:

			explicit NonStationary();
			~NonStationary();

			void ChangeEnv();

			float GetMu(int idx) const;

		protected:

			virtual void refresh_values(int action, float reward) override;

		private:

			float alpha_;
			int state_;
		};

	} // nonstationary
} // bandit

#endif // __NON_STATIONARY__
