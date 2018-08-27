#ifndef GROUP_H
#define GROUP_H

#include <string>
#include <vector>

using namespace std;

class group
{
public:
	string bagName;
	int grpIdx; //group index in the bag
	vector<double> center;
	vector<pair<int, int>> grpInsIdx; //contain index of all the instance belong to the group, first Idx is bag idx, second is ins idx in that bag
	int gGrpIdx; //global group index
	int ctrInsIdx; //record the grp ctr is which ins in the bag
};

#endif