#ifndef __LEARNING_FRAMEWORK__
#define __LEARNING_FRAMEWORK__

namespace native {

	class __declspec(dllexport) LearningFramework {

	public:

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
	};

} // native

#endif // __LEARNING_FRAMEWORK__
