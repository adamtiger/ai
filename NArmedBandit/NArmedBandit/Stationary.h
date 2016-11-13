#ifndef __STATIONARY__
#define __STATIONARY__

#include "Environment.h"


namespace bandit {
	namespace stationary {

		class Stationary : public BaseEnvironment {

		public:

			explicit Stationary();
			~Stationary();

		protected:

			virtual void refresh_values(int action, float reward) override;
		};

	} // stationary
} // bandit
#endif // __STATIONARY
