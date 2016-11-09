#ifndef __ACTION_SELECTOR__
#define __ACTION_SELECTOR__

namespace native {
	namespace alg {

		class ActionSelectors {

		public:

			static void EpsilonGreedy(int suggested, int* choosen);

			static void Softmax(int suggested, int* choosen);
		};

	} // alg
} // native

#endif // __ACTION_SELECTOR__
