#ifndef BAG_H
#define BAG_H

#include <iostream>
#include <vector>
#include <string>
#include "instance.h"
#include "group.h"

using namespace std;

class bag
{
public:
	string name;
	vector<instance> ins;
	int label; //0: non-musk, 1:musk
	vector<group> grp;//groups in this bag
	int index;
};

#endif