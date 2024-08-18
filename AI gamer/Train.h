#pragma once
#include "Viewer.h"
class Train{
public:
	Train();
	~Train();
	void TrainModel(size_t count, int width, int height, Viewer* viewer);
};