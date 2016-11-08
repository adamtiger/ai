#ifndef __LEARNING_FRAMEWORK_WRAPPER__
#define __LEARNING_FRAMEWORK_WRAPPER__

namespace native {

	class LearningFramework;

	namespace wrapper {

		public ref class LearningFrameworkWrapper{

		public:

			LearningFrameworkWrapper();
			~LearningFrameworkWrapper();
			!LearningFrameworkWrapper();

			void InitFramework(
				int type,                   // the type of the rl algorithm
				int rows, int columns,      // the size of the world
				int numofEpisodes,          // number of learning episodes
				int startX, int startY,     // coordinates of the starting cell
				int targetX, int targetY,   // coordinates of the target cell
				double alpha,               // the learning rate
				double gamma				// in case of discounted reward); 
			);

			void Learn();
			double GetProgress();

			// Copy the trace of the agent. 
			int GetPathLength();
			int GetEpisodeId();
			int GetCoordX(int idx);
			int GetCoordY(int idx);

		private:

			void Destroy();

			LearningFramework* lf;
		};

	} // wrapper
} // native


#endif // __LEARNING_FRAMEWORK_WRAPPER__

