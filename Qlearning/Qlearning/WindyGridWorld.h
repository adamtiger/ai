#ifndef __WINDY_GRID_WORLD__
#define __WINDY_GRID_WORLD__

#include <vector>

struct Position {

	explicit Position():
	x(0), y(0) {}

	Position(int x, int y) :
	x(x), y(y) {}

	Position(const Position& pos):
		x(pos.x), y(pos.y) {}

	int x;
	int y;

	void MoveWith(int dx, int dy) { x += dx; y += dy; }
	void MoveTo(int nx, int ny) { x = nx; y = ny; }
	void MoveTo(const Position& npos) { x = npos.x; y = npos.y; }

	bool operator==(const Position& other) {

		return (this->x == other.x && this->y == other.y);
	}
};


class WindyGridWorld {

public:

	explicit WindyGridWorld() :
		_start(0, 0), _goal(10, 10), _actual(_start),
		_height(10), _width(10), _numsteps(0)
	{		
	}

	WindyGridWorld(
		int height, int width, 
		int startx, int starty,
		int goalx, int goaly):
		_start(startx, starty), _goal(goalx, goaly),
		_height(height), _width(width), _numsteps(0)
	{
	}

	void Move(int dx, int dy) {

		++_numsteps;

		bool left, right, bottom, top;
		left = _actual.x + dx < 0 ? false : true;
		right = _actual.x + dx + 1 > _width ? false : true;
		bottom = _actual.y + dy < 0 ? false : true;
		top = _actual.y + dy + 1 > _height ? false : true;

		if (!(left && right && bottom && top)) {
			if (!left) { _actual.x = 0; _actual.y += dy; }
			else if (!bottom) { _actual.x += dx; _actual.y = 0; }
			else if (!right) { _actual.x = _width - 1; _actual.y += dy; }
			else { _actual.x += dx; _actual.y = _height - 1; }
		}
		else {
			_actual.MoveWith(dx, dy);
		}

		_traces.push_back(_actual);
	}

	void StartNewEpisode() {
		_actual.MoveTo(_start);
		_numsteps = 0;
		_traces.clear();
	}

	const Position& GetTraceAt(int time) {
		return _traces.at(time);
	}

	int LengthOfEpisode() {
		return _traces.size();
	}

	bool IsTerminated() {
		return (_start == _goal);
	}

private:

	int _height; // The sizes are expressed in the number of cells.
	int _width;

	Position _start;
	Position _actual; // The current position of the agent.
	Position _goal;

	std::vector<Position> _traces; // The path of the agent.

	int _numsteps;
};

#endif // __WINDY_GRID_WORLD__

