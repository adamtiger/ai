#ifndef __ENVIRONMENT__
#define __ENVIRONMENT__

#include "Logger.h"

using namespace std;

namespace native {
	namespace frw {

		struct Cell {

		public:

			Cell(int x, int y) :
				x(x), y(y) {}

			int x;
			int y;
		};

		class Environment {

		public:

			Environment(
				Logger* log,
				int rows, int cols,
				int startX, int startY,
				int targetX, int targetY);

			double RewardFunc() const;

			void ExecuteAction(int action);

		private:

			Logger* log_;

			int rows_;
			int cols_;
			Cell start_;
			Cell target_;

			Cell current_;

		};
	} // frw
} // native

#endif // __ENVIRONMENT__
