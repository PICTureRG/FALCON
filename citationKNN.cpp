#include "citationKNN.h"
#include <fstream>
#include <algorithm>
//#include <windows.h>
//#include "psapi.h"

long long countDistComp = 0;
long long countDistComp2 = 0;
long long countDistComp3 = 0;
long long countDistComp4 = 0;
long long countDistComp5 = 0;
int hDistType; //indicate which type of hausdorff dist used: max or min corresponding to 1 or 0
bool isOriginal; //if currently use original one, isOriginal = true, otherwise false;
double distSum = 0;
int distCount = 0;
double distCurBagToCiter;

void citationKNN::readData(string path)
{
	ifstream inFile;
	inFile.open(path);
	string line;

	int instIndex = 0;
	while(getline(inFile, line))
	{
		instance ins; 
		ins.gIdx = instIndex++;

		string bagName;
		int pos = line.find(",");
		bagName = line.substr(0, pos);
		ins.bagName = bagName;

		string insName;
		int pos2 = line.find(",", pos + 1);
		insName = line.substr(pos + 1, pos2 - pos - 1);
		ins.name = insName;

		do //extract attribute f1 to f166
		{
			pos = pos2;
			pos2 = line.find(",", pos + 1);
			if(pos2 == -1)
				break;
			string valStr = line.substr(pos + 1, pos2 - pos - 1);
			ins.attribute.push_back(stod(valStr));
		}while(true);

		pos2 = line.find(".", pos + 1); //extract the label of the instance, which ends with '.'
		string labelStr = line.substr(pos + 1, pos2 - pos - 1);
		ins.label = stoi(labelStr);

		if(hash.find(ins.bagName) != hash.end()) //bag has been created
			dataset[hash[ins.bagName]].ins.push_back(ins);
		else //create bag instance
		{
			dataset.push_back(bag()); //create a new bag instance and add to dataset
			dataset.rbegin()->index = dataset.size() - 1;
			dataset.rbegin()->name = ins.bagName; //name of new bag instance is from the instance's bagName
			hash[ins.bagName] = dataset.size() - 1; //build the map for the new bag
			dataset[hash[ins.bagName]].ins.push_back(ins);
			dataset[hash[ins.bagName]].label = ins.label;
		}
		//cout<<insName<<endl;
	}
	
	numInstances = instIndex;
}

bool isSame(vector<double> &a, vector<double> &b)
{
	for(int i = 0; i < a.size(); i++)
	{
		if(abs(a[i] - b[i]) > 1e-6)
		//if(abs(a[i] - b[i]) != 0)
			return false;
	}
	return true;
}

double dist2(vector<double> &a, vector<double> &b)
{
	double sum = 0;
	countDistComp2++;
	for(int i = 0; i < a.size(); i++)
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	
	return sqrt(sum);
}

double dist3(vector<double> &a, vector<double> &b)
{
	double sum = 0;
	countDistComp3++;
	for(int i = 0; i < a.size(); i++)
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	
	return sqrt(sum);
}

double dist4(vector<double> &a, vector<double> &b)
{
	double sum = 0;
	countDistComp4++;
	for(int i = 0; i < a.size(); i++)
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	
	return sqrt(sum);
}

double dist5(vector<double> &a, vector<double> &b)
{
	double sum = 0;
	countDistComp5++;
	for(int i = 0; i < a.size(); i++)
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	
	return sqrt(sum);
}

void citationKNN::kpp(int bagIdx, int k, int &cntGrpIdx)
{
	int nInst = dataset[bagIdx].ins.size();
	int nGrp = 0;
	double sum;
	vector<double> nrsDist(nInst, DBL_MAX);
	vector<int> nrsCtrIdx(nInst); //the index of nearest ctr

	group g;
	g.bagName = dataset[bagIdx].name;
	g.grpIdx = 0;
	int rdmIdx = rand() / (RAND_MAX - 1.) * nInst;
	g.center = dataset[bagIdx].ins[rdmIdx].attribute;
	g.gGrpIdx = cntGrpIdx++;
	g.ctrInsIdx = rdmIdx;
	dataset[bagIdx].grp.push_back(g);

	for(nGrp = 1; nGrp < k + 1; nGrp++)
	{
		sum = 0;
		double maxDist = -DBL_MAX;
		for(int i = 0; i < nInst; i++)
		{
			double ctrInsDist = dist(dataset[bagIdx].grp[nGrp - 1].center, dataset[bagIdx].ins[i].attribute);
			if(maxDist < ctrInsDist)
				maxDist = ctrInsDist;
			//dataset[bagIdx].ins[i].bagGrpCtrDist.push_back(make_pair(ctrInsDist, nGrp - 1));
			if(nrsDist[i] > ctrInsDist)
			{
				nrsDist[i] = ctrInsDist;
				nrsCtrIdx[i] = nGrp - 1;
			}
			sum += nrsDist[i];
		}
		int ctrInsIdx = dataset[bagIdx].grp[nGrp - 1].ctrInsIdx;
		dataset[bagIdx].ins[ctrInsIdx].furstInsDist = maxDist;
		if(nGrp == k) //last iteration, no need to get a new center
			break;
		sum = rand() / (RAND_MAX - 1.) * sum; //get a rdm num between 0 - sum
		for(int i = 0; i < nInst; i++)
		{
			if((sum -= nrsDist[i]) > 0)
				continue;
			group g;
			g.bagName = dataset[bagIdx].name;
			g.grpIdx = nGrp;
			g.center = dataset[bagIdx].ins[i].attribute; //use the instance's position of the first instance of every sizePerGrp instances as center
			g.gGrpIdx = cntGrpIdx++;
			g.ctrInsIdx = i;
			dataset[bagIdx].grp.push_back(g);
			break;
		}
	}

	for(int i1 = 0; i1 < nInst; i1++)
	{
		dataset[bagIdx].ins[i1].grpIdx = nrsCtrIdx[i1];
		dataset[bagIdx].ins[i1].grpCtrDist = nrsDist[i1];
		dataset[bagIdx].grp[nrsCtrIdx[i1]].grpInsIdx.push_back(make_pair(bagIdx, i1));
	}
}

void citationKNN::grouping(int numIter) //numIter: number of iteration
{
	int n = dataset.size();
	int cntGrpIdx = 0;
	for(int i = 0; i < n; i++)
	{
		int nInst = dataset[i].ins.size(); //number of instance in bag i
		int k;
		if(nInst % 10 == 0)
			k = nInst / 10;
		else
			k = nInst / 10 + 1;

		kpp(i, k, cntGrpIdx);
	}
	numGroups = cntGrpIdx;
}

void citationKNN::init()
{
	int n = dataset.size();
	distBtwBag = vector<vector<double>>(n, vector<double>(n, -1));
	curDistBtwBag = vector<vector<double>>(n, vector<double>(n, -1));
	//skpIns = vector<vector<vector<pair<pair<int, int>, double>>>>(n, vector<vector<pair<pair<int, int>, double>>>(n));
	//if(!isOriginal)
		//initBagInnerDist();
}

/*void citationKNN::initBagInnerDist()
{
	double maxDist;
	int furstInsIdx;
	for(int i = 0; i < dataset.size(); i++)
	{
		maxDist = 0;
		furstInsIdx = 0; //initialize as 0, because there could be only one instance in a bag
		for(int j = 1; j < dataset[i].ins.size(); j++) //cal distance between every instance (except the first one) to the first instance in the same bag
		{
			int d = dist3(dataset[i].ins[0].attribute, dataset[i].ins[j].attribute); 
			if(d > maxDist)
			{
				maxDist = d;
				furstInsIdx = j;
			}
		}
		dataset[i].ins[0].furstInsIdx = furstInsIdx;
		dataset[i].ins[0].furstInsDist = maxDist;
	}
}*/

double citationKNN::hausdorffDist(bag A, bag B, vector<vector<double>> &bagAInsBagBIns, vector<vector<double>> &distALmToBLm)
{
	vector<vector<double>> bagAGrpCtrBagBIns(A.grp.size(), vector<double>(B.ins.size(), -1));
	vector<vector<double>> bagBGrpCtrBagAIns(B.grp.size(), vector<double>(A.ins.size(), -1));

	if(hDistType == 0)
		return hxMin(A, B, 1, bagAGrpCtrBagBIns, bagBGrpCtrBagAIns, bagAInsBagBIns, distALmToBLm);
	else //hDistType == 1
		return hxMax(A, B, A.ins.size(), bagAGrpCtrBagBIns, bagBGrpCtrBagAIns, bagAInsBagBIns, distALmToBLm);
}

double citationKNN::hxMax(bag A, bag B, int k, vector<vector<double>> &bagAGrpCtrBagBIns, vector<vector<double>> &bagBGrpCtrBagAIns,
	vector<vector<double>> &bagAInsBagBIns, vector<vector<double>> &distALmToBLm)
{
	//multiset<double> minAtoB;
	double maxMinAtoB;
	if(curDistBtwBag[A.index][B.index] != -1)
		maxMinAtoB = curDistBtwBag[A.index][B.index];
	else
		maxMinAtoB = -DBL_MAX;

	for(int i = 0; i < A.ins.size(); i++)
	{
		double minInsToBag = DBL_MAX;
		bool overlook = false; //overlook is true means that current ins in B will not change the res finally returned, thus overlook this ins
		for(int j = 0; j < B.ins.size(); j++)
		{
			if(bagAInsBagBIns[i][j] == -1)
			{
				double lBound = -DBL_MAX;
				double uBound = DBL_MAX;
				double distBtwLm;
				double distToGrpCtr = A.ins[i].grpCtrDist;
				double distToGrpCtr2 = B.ins[j].grpCtrDist;
				int ctrInsIdx = A.grp[A.ins[i].grpIdx].ctrInsIdx; //get which ins the grp ctr is
				int ctrInsIdx2 = B.grp[B.ins[j].grpIdx].ctrInsIdx; //get which ins the grp ctr is

				if(bagAInsBagBIns[ctrInsIdx][ctrInsIdx2] != -1)
					distALmToBLm[A.ins[i].grpIdx][B.ins[j].grpIdx] = bagAInsBagBIns[ctrInsIdx][ctrInsIdx2];
				if(distALmToBLm[A.ins[i].grpIdx][B.ins[j].grpIdx] != -1)
					distBtwLm = distALmToBLm[A.ins[i].grpIdx][B.ins[j].grpIdx];
				else
					distBtwLm = bagAInsBagBIns[ctrInsIdx][ctrInsIdx2] = distALmToBLm[A.ins[i].grpIdx][B.ins[j].grpIdx] = dist4(A.grp[A.ins[i].grpIdx].center, B.grp[B.ins[j].grpIdx].center);
				lBound = max(lBound, distBtwLm - distToGrpCtr - distToGrpCtr2);
				uBound = min(uBound, distBtwLm + distToGrpCtr + distToGrpCtr2);
				
				if(uBound <= maxMinAtoB) //no need to cal, since it will not update the maxMinAtoB.
				{
					overlook = true;
					break;
				}
				if(lBound >= minInsToBag) //no need to cal dist between A.ins[i] and B.ins[j], since it will not update minInsToBag
					continue;

				double distBtwCtrIns;
				if(bagAInsBagBIns[ctrInsIdx][j] != -1)
					bagAGrpCtrBagBIns[A.ins[i].grpIdx][j] = bagAInsBagBIns[ctrInsIdx][j];
				if(bagAGrpCtrBagBIns[A.ins[i].grpIdx][j] != -1)
					distBtwCtrIns = bagAGrpCtrBagBIns[A.ins[i].grpIdx][j];
				else
					distBtwCtrIns = bagAInsBagBIns[ctrInsIdx][j] = bagAGrpCtrBagBIns[A.ins[i].grpIdx][j] = dist4(A.grp[A.ins[i].grpIdx].center, B.ins[j].attribute);

				lBound = max(lBound, abs(distBtwCtrIns - distToGrpCtr));
				uBound = min(uBound, distBtwCtrIns + distToGrpCtr);
				if(uBound <= maxMinAtoB) //no need to cal, since it will not update the maxMinAtoB.
				{
					overlook = true;
					break;
				}
				if(lBound >= minInsToBag) //no need to cal dist between A.ins[i] and B.ins[j], since it will not update minInsToBag
					continue;

				double distBtwCtrIns2;
				if(bagAInsBagBIns[i][ctrInsIdx2] != -1)
					bagBGrpCtrBagAIns[B.ins[j].grpIdx][i] = bagAInsBagBIns[i][ctrInsIdx2];
				if(bagBGrpCtrBagAIns[B.ins[j].grpIdx][i] != -1)
					distBtwCtrIns2 = bagBGrpCtrBagAIns[B.ins[j].grpIdx][i];
				else
					distBtwCtrIns2 = bagAInsBagBIns[i][ctrInsIdx2] = bagBGrpCtrBagAIns[B.ins[j].grpIdx][i] = dist4(B.grp[B.ins[j].grpIdx].center, A.ins[i].attribute);

				lBound = max(lBound, abs(distBtwCtrIns2 - distToGrpCtr2));
				uBound = min(uBound, distBtwCtrIns2 + distToGrpCtr2);
				if(uBound <= maxMinAtoB) //no need to cal, since it will not update the maxMinAtoB.
				{
					overlook = true;
					break;
				}
				if(lBound >= minInsToBag) //no need to cal dist between A.ins[i] and B.ins[j], since it will not update minInsToBag
					continue;
			}
			double insDist;
			if(bagAInsBagBIns[i][j] != -1)
				insDist = bagAInsBagBIns[i][j];
			else //no existing value
				insDist = bagAInsBagBIns[i][j] = dist5(A.ins[i].attribute, B.ins[j].attribute);
			if(insDist <= maxMinAtoB)
			{
				overlook = true;
				break;
			}
			minInsToBag = min(minInsToBag, insDist);
		}
		
		if(overlook)
			continue;
		maxMinAtoB = max(maxMinAtoB, minInsToBag);
		if(maxMinAtoB != DBL_MAX && maxMinAtoB >= distCurBagToCiter)
		{
			curDistBtwBag[A.index][B.index] = curDistBtwBag[B.index][A.index] = maxMinAtoB;
			return -1;
		}

	}

	for(int i = 0; i < B.ins.size(); i++)
	{
		double res = maxMinAtoB;
		double minInsToBag = DBL_MAX;
		bool overlook = false; //overlook is true means that current ins in B will not change the res finally returned, thus overlook this ins
		for(int j = 0; j < A.ins.size(); j++)
		{
			if(bagAInsBagBIns[j][i] != -1) //has been cal
			{
				if(bagAInsBagBIns[j][i] <= res)
				{
					overlook = true;
					break;
				}
				minInsToBag = min(minInsToBag, bagAInsBagBIns[j][i]);
			}
		}
		if(overlook)
			continue;
		for(int j = 0; j < A.ins.size(); j++)
		{
			if(bagAInsBagBIns[j][i] == -1) //has not been cal
			{
				double lBound = -DBL_MAX;
				double uBound = DBL_MAX;
				double distBtwLm;
				double distToGrpCtr = B.ins[i].grpCtrDist;
				double distToGrpCtr2 = A.ins[j].grpCtrDist;
				int ctrInsIdx = B.grp[B.ins[i].grpIdx].ctrInsIdx; //get which ins the grp ctr is
				int ctrInsIdx2 = A.grp[A.ins[j].grpIdx].ctrInsIdx; //get which ins the grp ctr is

				if(bagAInsBagBIns[ctrInsIdx2][ctrInsIdx] != -1)
					distALmToBLm[A.ins[j].grpIdx][B.ins[i].grpIdx] = bagAInsBagBIns[ctrInsIdx2][ctrInsIdx];
				if(distALmToBLm[A.ins[j].grpIdx][B.ins[i].grpIdx] != -1)
					distBtwLm = distALmToBLm[A.ins[j].grpIdx][B.ins[i].grpIdx];
				else
					distBtwLm = bagAInsBagBIns[ctrInsIdx2][ctrInsIdx] = distALmToBLm[A.ins[j].grpIdx][B.ins[i].grpIdx] = dist4(A.grp[A.ins[j].grpIdx].center, B.grp[B.ins[i].grpIdx].center);
				lBound = max(lBound, distBtwLm - distToGrpCtr - distToGrpCtr2);
				uBound = min(uBound, distBtwLm + distToGrpCtr + distToGrpCtr2);

				if(uBound <= res)
				{
					overlook = true;
					break;
				}
				if(lBound >= minInsToBag) //no need to cal dist between A.ins[i] and B.ins[j], since it will not update minInsToBag
					continue;

				double distBtwCtrIns;
				if(bagBGrpCtrBagAIns[B.ins[i].grpIdx][j] != -1)
					distBtwCtrIns = bagBGrpCtrBagAIns[B.ins[i].grpIdx][j];
				else
					distBtwCtrIns = bagBGrpCtrBagAIns[B.ins[i].grpIdx][j] = dist4(B.grp[B.ins[i].grpIdx].center, A.ins[j].attribute);
				
				bagAInsBagBIns[j][ctrInsIdx] = distBtwCtrIns;

				lBound = max(lBound, abs(distBtwCtrIns - distToGrpCtr));
				uBound = min(uBound, distBtwCtrIns + distToGrpCtr);
				if(uBound <= res)
				{
					overlook = true;
					break;
				}
				if(lBound >= minInsToBag) //no need to cal dist between A.ins[i] and B.ins[j], since it will not update minInsToBag
					continue;

				double distBtwCtrIns2;
				if(bagAGrpCtrBagBIns[A.ins[j].grpIdx][i] != -1)
					distBtwCtrIns2 = bagAGrpCtrBagBIns[A.ins[j].grpIdx][i];
				else
					distBtwCtrIns2 = bagAGrpCtrBagBIns[A.ins[j].grpIdx][i] = dist4(A.grp[A.ins[j].grpIdx].center, B.ins[i].attribute);
				
				bagAInsBagBIns[ctrInsIdx2][i] = distBtwCtrIns2;

				lBound = max(lBound, abs(distBtwCtrIns2 - distToGrpCtr2));
				uBound = min(uBound, distBtwCtrIns2 + distToGrpCtr2);
				if(uBound <= res)
				{
					overlook = true;
					break;
				}
				if(lBound >= minInsToBag) //no need to cal dist between A.ins[i] and B.ins[j], since it will not update minInsToBag
					continue;
			}
			double insDist;
			if(bagAInsBagBIns[j][i] != -1)
				insDist = bagAInsBagBIns[j][i];
			else
				insDist = bagAInsBagBIns[j][i] = dist5(B.ins[i].attribute, A.ins[j].attribute);
			if(insDist <= res)
			{
				overlook = true;
				break;
			}
			minInsToBag = min(minInsToBag, insDist);
		}
		if(overlook)
			continue;
		maxMinAtoB = max(maxMinAtoB, minInsToBag);
		if(maxMinAtoB != DBL_MAX && maxMinAtoB >= distCurBagToCiter)
		{
			curDistBtwBag[A.index][B.index] = curDistBtwBag[B.index][A.index] = maxMinAtoB;
			return -1;
		}
	}

	return maxMinAtoB;
}

double citationKNN::hxMin(bag A, bag B, int k, vector<vector<double>> &bagAGrpCtrBagBIns, vector<vector<double>> &bagBGrpCtrBagAIns,
	vector<vector<double>> &bagAInsBagBIns, vector<vector<double>> &distALmToBLm)  //k is the k-th ranked distance
{
	multiset<double> minAtoB;

	if(curDistBtwBag[A.index][B.index] != -1)
	{
		if(distCurBagToCiter != DBL_MAX && curDistBtwBag[A.index][B.index] < distCurBagToCiter)
			return -2;
		minAtoB.insert(curDistBtwBag[A.index][B.index]);
	}
	else
		minAtoB.insert(DBL_MAX);
	bool realDist = true;
	for(int i = 0; i < A.ins.size(); i++)
	{
		double minInsToBag = DBL_MAX;
		for(int j = 0; j < B.ins.size(); j++)
		{			
			if(bagAInsBagBIns[i][j] == -1)
			{
				double lBound = -DBL_MAX;
				double uBound = DBL_MAX;
				double distBtwLm;
				double distToGrpCtr = A.ins[i].grpCtrDist;
				double distToGrpCtr2 = B.ins[j].grpCtrDist;
				int ctrInsIdx = A.grp[A.ins[i].grpIdx].ctrInsIdx; //get which ins the grp ctr is;
				int ctrInsIdx2 = B.grp[B.ins[j].grpIdx].ctrInsIdx; //get which ins the grp ctr is
				
				if(bagAInsBagBIns[ctrInsIdx][ctrInsIdx2] != -1)
					distALmToBLm[A.ins[i].grpIdx][B.ins[j].grpIdx] = bagAInsBagBIns[ctrInsIdx][ctrInsIdx2];
				if(distALmToBLm[A.ins[i].grpIdx][B.ins[j].grpIdx] != -1)
					distBtwLm = distALmToBLm[A.ins[i].grpIdx][B.ins[j].grpIdx];
				else
					distBtwLm = bagAInsBagBIns[ctrInsIdx][ctrInsIdx2] = distALmToBLm[A.ins[i].grpIdx][B.ins[j].grpIdx] = dist4(A.grp[A.ins[i].grpIdx].center, B.grp[B.ins[j].grpIdx].center);
				lBound = max(lBound, distBtwLm - distToGrpCtr - distToGrpCtr2);
				uBound = distBtwLm + distToGrpCtr + distToGrpCtr2;

				if(distCurBagToCiter != DBL_MAX && uBound < distCurBagToCiter)
				{
					curDistBtwBag[A.index][B.index] = curDistBtwBag[B.index][A.index] = *minAtoB.rbegin();
					return -2;
				}
				if(lBound >= minInsToBag) //no need to cal dist between A.ins[i] and B.ins[j], since it will not update minInsToBag
					continue;
				if(minAtoB.size() == k && lBound >= *minAtoB.rbegin()) //no need to cal, since it will not update the kth element in minAtoB.
					continue;

				double distBtwCtrIns;
				
				if(bagAInsBagBIns[ctrInsIdx][j] != -1)
					bagAGrpCtrBagBIns[A.ins[i].grpIdx][j] = bagAInsBagBIns[ctrInsIdx][j];
				if(bagAGrpCtrBagBIns[A.ins[i].grpIdx][j] !=-1)
					distBtwCtrIns = bagAGrpCtrBagBIns[A.ins[i].grpIdx][j];
				else
					distBtwCtrIns = bagAInsBagBIns[ctrInsIdx][j] = bagAGrpCtrBagBIns[A.ins[i].grpIdx][j] = dist4(A.grp[A.ins[i].grpIdx].center, B.ins[j].attribute);
				
				lBound = abs(distBtwCtrIns - distToGrpCtr);
				uBound = distBtwCtrIns + distToGrpCtr;
				if(distCurBagToCiter != DBL_MAX && uBound < distCurBagToCiter)
				{
					curDistBtwBag[A.index][B.index] = curDistBtwBag[B.index][A.index] = *minAtoB.rbegin();
					return -2;
				}
				if(lBound >= minInsToBag) //no need to cal dist between A.ins[i] and B.ins[j], since it will not update minInsToBag
					continue;
				if(minAtoB.size() == k && lBound >= *minAtoB.rbegin()) //no need to cal, since it will not update the kth element in minAtoB.
					continue;

				double distBtwCtrIns2;
				if(bagAInsBagBIns[i][ctrInsIdx2] != -1)
					bagBGrpCtrBagAIns[B.ins[j].grpIdx][i] = bagAInsBagBIns[i][ctrInsIdx2];
				if(bagBGrpCtrBagAIns[B.ins[j].grpIdx][i] != -1)
					distBtwCtrIns2 = bagBGrpCtrBagAIns[B.ins[j].grpIdx][i];
				else
					distBtwCtrIns2 = bagAInsBagBIns[i][ctrInsIdx2] = bagBGrpCtrBagAIns[B.ins[j].grpIdx][i] = dist4(B.grp[B.ins[j].grpIdx].center, A.ins[i].attribute);
				
				lBound = max(lBound, abs(distBtwCtrIns2 - distToGrpCtr2));
				uBound = min(uBound, distBtwCtrIns2 + distToGrpCtr2);
				if(distCurBagToCiter != DBL_MAX && uBound < distCurBagToCiter)
				{
					curDistBtwBag[A.index][B.index] = curDistBtwBag[B.index][A.index] = *minAtoB.rbegin();
					return -2;
				}
				if(lBound >= minInsToBag) //no need to cal dist between A.ins[i] and B.ins[j], since it will not update minInsToBag
					continue;
				if(minAtoB.size() == k && lBound >= *minAtoB.rbegin()) //no need to cal, since it will not update the kth element in minAtoB.
					continue;
				if(lBound >= distCurBagToCiter)
				{
					//skpIns[A.index][B.index].push_back(make_pair(make_pair(i, j), lBound));
					realDist = false;
					continue;
				}
			}
			double insDist;
			if(bagAInsBagBIns[i][j] != -1)
				insDist = bagAInsBagBIns[i][j];
			else //no existing value
				insDist = bagAInsBagBIns[i][j] = dist5(A.ins[i].attribute, B.ins[j].attribute);

			if(distCurBagToCiter != DBL_MAX && insDist < distCurBagToCiter)
				return -2;

			minInsToBag = min(minInsToBag, insDist);
		}
		minAtoB.insert(minInsToBag);
		if(minAtoB.size() > k)
			minAtoB.erase(prev(minAtoB.end()));
	}
	if(*minAtoB.rbegin() > distCurBagToCiter && !realDist) //has skipped some ins, and min dis < distCurBagToCiter, thus skipped ins could have real min dis
	{
		curDistBtwBag[A.index][B.index] = curDistBtwBag[B.index][A.index] = *minAtoB.rbegin();
		return -1;
	}

	return *minAtoB.rbegin(); //return the last element in the multiset
}

double citationKNN::dist(vector<double> &a, vector<double> &b)
{
	double sum = 0;
	countDistComp++;
	for(int i = 0; i < a.size(); i++)
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	
	return sqrt(sum);
}


bool cmp(const pair<double, string> &a, const pair<double, string> &b)
{
    if(a.first <= b.first)
        return true;
    return false;
}

void citationKNN::clean(int bagIdx)
{
	int n = dataset.size();
	for(int i = 0; i < n; i++)
	{
		distBtwBag[i][bagIdx] = distBtwBag[bagIdx][i] = -1;
		for(int j = 0; j < n; j++)
		{
			curDistBtwBag[i][j] = -1;
			//skpIns[i][j].clear();
		}
	}
}

int citationKNN::predict(int curBagIdx, int R, int C) //predict the label of test bag (curBagIdx), R is the number of reference, C is the C in C-nearest citer
{	
	vector<int> vote; //labels of all reference and citers.
	int n = dataset.size();
	for(int i = 0; i < dataset.size(); i++)
	{
		//cout<<i<<endl;
		vector<int> bagOrder;
		for(int j = 0; j < dataset.size(); j++)
			bagOrder.push_back(j);
		swap(bagOrder[0], bagOrder[curBagIdx]);
		int i1 = 0;
		for(int i2 = 1; i2 < n; i2++)
		{  
			if(distBtwBag[i][bagOrder[i2]] != -1)
				swap(bagOrder[++i1], bagOrder[i2]);
		}

		multiset<pair<double, int>> distBag; //double is for the Hausdorff dist, int is the index of bag
		bool skip = false; //if skip is true, current bag will not be citer
		distCurBagToCiter = DBL_MAX;
		for(auto j : bagOrder)
		{		
			if(i == j)
				continue;
			vector<vector<double>> bagAInsBagBIns(dataset[i].ins.size(), vector<double>(dataset[j].ins.size(), -1)); //hash table for instance dist between bagA and bagB
			vector<vector<double>> distALmToBLm(dataset[i].grp.size(), vector<double>(dataset[j].grp.size(), -1)); //dist btw landmarks in A and B
			double hDist;
			if(distBtwBag[i][j] != -1)
				hDist = distBtwBag[i][j];
			else 
			{
				double lBound = 0.0, uBound = DBL_MAX;
				if(hDistType == 0)
				{			
					int ctrInsIdx1 = dataset[i].grp[0].ctrInsIdx;
					int ctrInsIdx2 = dataset[j].grp[0].ctrInsIdx;
					distALmToBLm[0][0] = dist3(dataset[i].ins[ctrInsIdx1].attribute, dataset[j].ins[ctrInsIdx2].attribute);
					//distBtwLm = dist3(dataset[i].ins[ctrInsIdx1].attribute, dataset[j].ins[ctrInsIdx2].attribute);
					bagAInsBagBIns[ctrInsIdx1][ctrInsIdx2] = distALmToBLm[0][0];
					lBound = max(lBound, distALmToBLm[0][0] - dataset[i].ins[ctrInsIdx1].furstInsDist - dataset[j].ins[ctrInsIdx2].furstInsDist);
					uBound = min(uBound, distALmToBLm[0][0] + dataset[i].ins[ctrInsIdx1].furstInsDist + dataset[j].ins[ctrInsIdx2].furstInsDist);
					

					/*for(int k1 = 0; k1 < dataset[i].grp.size(); k1++)
					{
						int ctrInsIdx1 = dataset[i].grp[k1].ctrInsIdx;
						for(int t = 0; t < min((int)dataset[j].grp.size(), 2); t++)
						{
							int k2 = (k1 + t) % dataset[j].grp.size();
							int ctrInsIdx2 = dataset[j].grp[k2].ctrInsIdx;
							distALmToBLm[k1][k2] = dist3(dataset[i].ins[ctrInsIdx1].attribute, dataset[j].ins[ctrInsIdx2].attribute);
							bagAInsBagBIns[ctrInsIdx1][ctrInsIdx2] = distALmToBLm[k1][k2];
							lBound = max(lBound, distALmToBLm[k1][k2] - dataset[i].ins[ctrInsIdx1].furstInsDist - dataset[j].ins[ctrInsIdx2].furstInsDist);
							uBound = min(uBound, distALmToBLm[k1][k2] + dataset[i].ins[ctrInsIdx1].furstInsDist + dataset[j].ins[ctrInsIdx2].furstInsDist);
						}
					}*/
					/*for(int k1 = 0; k1 < min(dataset[i].grp.size(), dataset[j].grp.size()); k1++)
					{
						int ctrInsIdx1 = dataset[i].grp[k1].ctrInsIdx;vim
						int ctrInsIdx2 = dataset[j].grp[k1].ctrInsIdx;
						distALmToBLm[k1][k1] = dist3(dataset[i].ins[ctrInsIdx1].attribute, dataset[j].ins[ctrInsIdx2].attribute);
						bagAInsBagBIns[ctrInsIdx1][ctrInsIdx2] = distALmToBLm[k1][k1];
						lBound = max(lBound, distALmToBLm[k1][k1] - dataset[i].ins[ctrInsIdx1].furstInsDist - dataset[j].ins[ctrInsIdx2].furstInsDist);
						uBound = min(uBound, distALmToBLm[k1][k1] + dataset[i].ins[ctrInsIdx1].furstInsDist + dataset[j].ins[ctrInsIdx2].furstInsDist);
					}*/

					/*for(int k1 = 0; k1 < dataset[i].grp.size(); k1++)
					{
						int ctrInsIdx1 = dataset[i].grp[k1].ctrInsIdx;
						for(int k2 = 0; k2 <dataset[j].grp.size(); k2++)
						{
							int ctrInsIdx2 = dataset[j].grp[k2].ctrInsIdx;
							distALmToBLm[k1][k2] = dist3(dataset[i].ins[ctrInsIdx1].attribute, dataset[j].ins[ctrInsIdx2].attribute);
							bagAInsBagBIns[ctrInsIdx1][ctrInsIdx2] = distALmToBLm[k1][k2];
							lBound = max(lBound, distALmToBLm[k1][k2] - dataset[i].ins[ctrInsIdx1].furstInsDist - dataset[j].ins[ctrInsIdx2].furstInsDist);
							uBound = min(uBound, distALmToBLm[k1][k2] + dataset[i].ins[ctrInsIdx1].furstInsDist + dataset[j].ins[ctrInsIdx2].furstInsDist);
						}
					}*/
				}
				else
				{
					int ctrInsIdx1 = dataset[i].grp[0].ctrInsIdx;
					int ctrInsIdx2 = dataset[j].grp[0].ctrInsIdx;
					distALmToBLm[0][0] = dist3(dataset[i].ins[ctrInsIdx1].attribute, dataset[j].ins[ctrInsIdx2].attribute);
					bagAInsBagBIns[ctrInsIdx1][ctrInsIdx2] = distALmToBLm[0][0];
					lBound = max(lBound, distALmToBLm[0][0] - min(dataset[i].ins[ctrInsIdx1].furstInsDist, dataset[j].ins[ctrInsIdx2].furstInsDist));
					/*for(int k1 = 0; k1 < dataset[i].grp.size(); k1++)
					{
						int ctrInsIdx1 = dataset[i].grp[k1].ctrInsIdx;
						for(int k2 = 0; k2 <dataset[j].grp.size(); k2++)
						{
							int ctrInsIdx2 = dataset[j].grp[k2].ctrInsIdx;
							distALmToBLm[k1][k2] = dist3(dataset[i].ins[ctrInsIdx1].attribute, dataset[j].ins[ctrInsIdx2].attribute);
							bagAInsBagBIns[ctrInsIdx1][ctrInsIdx2] =distALmToBLm[k1][k2];
							lBound = max(lBound, distALmToBLm[k1][k2] - min(dataset[i].ins[ctrInsIdx1].furstInsDist, dataset[j].ins[ctrInsIdx2].furstInsDist));
						}
					}*/
					/*for(int k1 = 0; k1 < dataset[i].grp.size(); k1++)
					{
						int ctrInsIdx1 = dataset[i].grp[k1].ctrInsIdx;
						for(int t = 0; t < min((int)dataset[j].grp.size(), 2); t++)
						{
							int k2 = (k1 + t) % dataset[j].grp.size();
							int ctrInsIdx2 = dataset[j].grp[k2].ctrInsIdx;
							distALmToBLm[k1][k2] = dist3(dataset[i].ins[ctrInsIdx1].attribute, dataset[j].ins[ctrInsIdx2].attribute);
							bagAInsBagBIns[ctrInsIdx1][ctrInsIdx2] = distALmToBLm[k1][k2];
							lBound = max(lBound, distALmToBLm[k1][k2] - min(dataset[i].ins[ctrInsIdx1].furstInsDist, dataset[j].ins[ctrInsIdx2].furstInsDist));
						}
					}*/
					/*for(int k1 = 0; k1 < min(dataset[i].grp.size(), dataset[j].grp.size()); k1++)
					{
						int ctrInsIdx1 = dataset[i].grp[k1].ctrInsIdx;
						int ctrInsIdx2 = dataset[j].grp[k1].ctrInsIdx;
						distALmToBLm[k1][k1] = dist3(dataset[i].ins[ctrInsIdx1].attribute, dataset[j].ins[ctrInsIdx2].attribute);
						bagAInsBagBIns[ctrInsIdx1][ctrInsIdx2] = distALmToBLm[k1][k1];
						lBound = max(lBound, distALmToBLm[k1][k1] - min(dataset[i].ins[ctrInsIdx1].furstInsDist, dataset[j].ins[ctrInsIdx2].furstInsDist));
					}*/
				}
				if(i == curBagIdx && (R == 0 || distBag.size() == R && lBound >= (*distBag.rbegin()).first))
					continue;
				if(i != curBagIdx && (C == 0 || distBag.size() == C && lBound >= (*distBag.rbegin()).first))
					continue;
				//if(i != curBagIdx && (C == 0 || lBound > distCurBagToCiter))
					//continue;

				hDist = distBtwBag[i][j] = distBtwBag[j][i] = hausdorffDist(dataset[i], dataset[j], bagAInsBagBIns, distALmToBLm);
				
				if(hDist == -1) //go to next ins directly
					continue;
				if(hDist == -2) //dist btw i and j is uncertain, but we know this bag will be in front of curBag (curBagIdx)
				{
					distBtwBag[i][j] = distBtwBag[j][i] = -1;
					hDist = 0; 
				}
			}

			if(j == curBagIdx)
				distCurBagToCiter = hDist; //dist between curbag (curBagIdx) and bag i

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
				{
					//cout<<k.second<<" "<<dataset[k.second].label<<"xxx"<<endl;
					vote.push_back(dataset[k.second].label);
				}
				else if(k.second == curBagIdx) //bag of bagIdx is in the C-nearest neighbors of bag of i
				{
					//cout<<i<<" "<<dataset[i].label<<"zzz"<<endl;
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

	return voteMusk > (vote.size() / 2);  //if tie, return 0*/
}
