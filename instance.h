#ifndef INSTNACE_H
#define INSTANCE_H

#include <vector>
#include <string>
using namespace std;

class instance
{
public:
	string bagName;
	string name;
	vector<double> attribute;
	int label; //0: non-musk, 1:musk
	int gIdx; //global index of the instance in the whole dataset
	int furstInsIdx; //index of furthest ins in the same bag
	double furstInsDist; //dist of furthest ins in the same bag
	int grpIdx; //group index of the instance in the bag
	double grpCtrDist; //distance to the group center
	//vector<pair<double, int>> bagGrpCtrDist; //distance to all center in the same bag, double is dist, int is index 
};

#endif
