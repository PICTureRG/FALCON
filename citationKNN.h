#ifndef CITATION_KNN_H
#define CITATION_KNN_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <limits.h>
#include <set>
#include <cfloat>
#include <queue>
#include "bag.h"
#include "group.h"

using namespace std;

class citationKNN
{
public:
	vector<bag> dataset;
	unordered_map<string, int> hash; //string is the name of bag, int is the index to the bag instance in dataset
	int numInstances; //the number of instances in the dataset 
	int numGroups; //the number of groups in the dataset
	vector<vector<double>> distBtwBag; //distance between bags
	vector<vector<double>> curDistBtwBag; //cur distance between bags but it could be not the real distance between bags due to rule3 filtering possible dist cal between ins

	void readData(string path);
	double hausdorffDist(bag A, bag B, vector<vector<double>> &bagAInsBagBIns, vector<vector<double>> &distALmToBLm);
	double hausdorffDist(bag A, bag B);
	double ori_hausdorffDist(bag A, bag B);
	double hxMin(bag A, bag B, int k, vector<vector<double>> &bagAGrpCtrBagBIns, vector<vector<double>> &bagBGrpCtrBagAIns,
	vector<vector<double>> &bagAInsBagBIns, vector<vector<double>> &distALmToBLm); //hx used for optimized CKNN
	double ori_hx(bag A, bag B, int k, vector<vector<double>> &bagAInsBagBIns, int type); //hx used for original CKNN
	double hxMax(bag A, bag B, int k, vector<vector<double>> &bagAGrpCtrBagBIns, vector<vector<double>> &bagBGrpCtrBagAIns,
	vector<vector<double>> &bagAInsBagBIns, vector<vector<double>> &distALmToBLm);
	double dist(vector<double> &a, vector<double> &b); //Use for general purpose Euclid distance between two vector
	int predict(int bagIndex, int R, int C); 
	int ori_predict(int bagIndex, int R, int C); 
	void initBagInnerDist(); //cal distance between every instance (except the first one) to the first instance in the same bag
	void init(); //init data structures used
	void grouping(int numIter); //in each bag, grouping instances into different groups
	void grouping(int k, int numLimit, int numIter); //k is the number of branch, when instance in a group <= numLimit, stop further grouping, numIter is the number of iterations
	void hirarchicalGrouping(int k, int numLimit, int numIter, vector<pair<int, int>> &grpData, int &countGroupIndex);
	void kpp(int bagIdx, int k, int &cntGrpIdx); //Kmeans++, k is the number of groups, bagIdx indicates which bag it processes, cntGrpIdx is to record which grp it is in the global scene
	//void kpp(int k, vector<pair<int, int>> &grpData, vector<group> &hrchGrp); //same function as the one above, used for hierarchical k-means
	void clean(int bagIdx); //clean existing info of specified bagIdx so that program predicts it as a new bag
};

#endif