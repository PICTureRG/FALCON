#include "citationKNN.h"
#include <fstream>
#include <algorithm>

extern int hDistType;

int citationKNN::ori_predict(int curBagIdx, int R, int C)
{
	vector<int> vote; //labels of all reference and citers.
	int n = dataset.size();
	for(int i = 0; i < dataset.size(); i++)
	{
		multiset<pair<double, int>> distBag; //double is for the Hausdorff dist, int is the index of bag
		bool skip = false;
		for(int j1 = 0; j1 < dataset.size(); j1++)
		{
			int j = j1;
			if(j1 == 0)
				j = curBagIdx;
			else if(j1 == curBagIdx)
				j = 0;

			if(i == j)
				continue;
			double hDist;
			if(distBtwBag[i][j] != -1)
				hDist = distBtwBag[i][j];
			else
				hDist = distBtwBag[i][j] = distBtwBag[j][i] = ori_hausdorffDist(dataset[i], dataset[j]);
			distBag.insert(make_pair(hDist, j));
			if((i == curBagIdx && distBag.size() > R) || (i != curBagIdx && distBag.size() > C))
			{
				if(distBag.rbegin()->second == curBagIdx)
				{
					skip = true;
					break;
				}
				distBag.erase(prev(distBag.end()));
			}
		}
		if(!skip)
		{
			for(auto k : distBag)
			{
				if(i == curBagIdx) //current bag is the query bag
					vote.push_back(dataset[k.second].label);
				else if(k.second == curBagIdx) //bag of bagIdx is in the C-nearest neighbors of bag of i
				{
					vote.push_back(dataset[i].label);
					break;
				}
			}
		}
	}
	int voteMusk = 0;
	for(int i = 0; i < vote.size(); i++)
	{
		if(vote[i] == 1)
			voteMusk++;
	}

	return voteMusk > (vote.size() / 2);  //if tie, return 0
}

double citationKNN::ori_hausdorffDist(bag A, bag B)
{
	vector<vector<double>> bagAInsBagBIns(A.ins.size(), vector<double>(B.ins.size(), -1)); //hash table for instance dist between bagA and bagB
	
	if(hDistType == 0) 
		return ori_hx(A, B, 1, bagAInsBagBIns, 0);
	else //hDistType == 1
		return max(ori_hx(A, B, A.ins.size(), bagAInsBagBIns, 0), ori_hx(B, A, B.ins.size(), bagAInsBagBIns, 1));
}

double citationKNN::ori_hx(bag A, bag B, int k, vector<vector<double>> &bagAInsBagBIns, int type)
{
	multiset<double> minAtoB;
	for(int i = 0; i < A.ins.size(); i++)
	{
		double minInsToBag = DBL_MAX; //minInsToBag is the min distance from A.ins[i] to bag B.
		for(int j = 0; j < B.ins.size(); j++)
		{
			double insDist ;
			if(type == 0)
			{
				if(bagAInsBagBIns[i][j] != -1)
					insDist = bagAInsBagBIns[i][j];
				else //no existing value
					insDist = bagAInsBagBIns[i][j] = dist(A.ins[i].attribute, B.ins[j].attribute);
			}
			else
			{
				if(bagAInsBagBIns[j][i] != -1)
					insDist = bagAInsBagBIns[j][i];
				else // no existing balue
					insDist = bagAInsBagBIns[j][i] = dist(A.ins[i].attribute, B.ins[j].attribute);
			}
			minInsToBag = min(minInsToBag, insDist);
		}
		minAtoB.insert(minInsToBag);
		if(minAtoB.size() > k)
			minAtoB.erase(prev(minAtoB.end()));
	}

	return *minAtoB.rbegin(); //return the last element in the multiset
}