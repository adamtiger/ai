#ifndef __ENVIRONMENT__
#define __ENVIRONMENT__

#include <vector>

using namespace std;

namespace native {
	namespace frw {

		struct Cell {

		public:

			Cell(int x = 0, int y= 0) :
				x(x), y(y) {}

			int x;
			int y;

			bool operator==(const Cell& o) const{
				return (this->x == o.x) && (this->y == o.y); 
			}
		};

		class Environment {

		public:

			explicit Environment();
			Environment(
				int rows, int cols,
				int startX, int startY,
				int targetX, int targetY);

			double RewardFunc() const;

			void ExecuteAction(int action);

			int GetX() const;
			int GetY() const;

		private:

			void restrict2gridworld(Cell& candidate);

			int rows_;
			int cols_;
			Cell start_;
			Cell target_;

			Cell current_;

			vector<int> wind_strength_;
		};

	} // frw
} // native

#endif // __ENVIRONMENT__
