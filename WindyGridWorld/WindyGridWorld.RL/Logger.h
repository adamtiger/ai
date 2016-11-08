#ifndef __LOGGER__
#define __LOGGER__

#include <vector>

using namespace std;

namespace native {
	namespace frw {

		class Logger {

		private:
			Logger();

		public:

			static Logger* Instance();

			void Activate(int epsId);
			void Passivate();

			void Add(int x, int y);

			int GetEpisodeId() const;
			int GetPathLength() const;
			int GetXcoord(int idx) const;
			int GetYcoord(int idx) const;

		private:
			static Logger* instance_;

			bool is_active_;
			int epsId_;
			vector<int> path_x_coords_;
			vector<int> path_y_coords_;
		};

	} // frw
} // native
#endif // __LOGGER__
