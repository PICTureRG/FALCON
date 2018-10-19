#include <iostream>
#include <string>
#include <vector>
#include "citationKNN.h"
#include <ctime>
#include <fstream>

using namespace std;

extern long long countDistComp;
extern long long countDistComp2;
extern long long countDistComp3;
extern long long countDistComp4;
extern long long countDistComp5;
extern int hDistType;
extern bool isOriginal;
extern double distSum;
extern int distCount;

int main()
{
	citationKNN CKNN;
	//CKNN.readData("harddrive1.csv");
	//CKNN.readData("protein.csv");
	//CKNN.readData("corel.csv");
	//CKNN.readData("bc.csv");
	//CKNN.readData("output.csv");
	CKNN.readData("clean2.data");
	//CKNN.readData("webRecom2.csv");
	hDistType = 0;
	isOriginal = false;
	int R = 2;
	int C = 4;
	int start_s=clock();
	if(!isOriginal)
		CKNN.grouping(1);
	CKNN.init();
	vector<int> res;
	for(int i = 0; i < CKNN.dataset.size(); i++)
	//for(int i = 0; i < 1; i++)
	{
		cout<<"Pkg: "<<i<<endl;
		CKNN.clean(i); //clean all existing info about i
		if(!isOriginal)
			res.push_back(CKNN.predict(i, R, C));
		else
			res.push_back(CKNN.ori_predict(i, R, C));
	}
	int stop_s=clock();

	cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << endl;
	
	for(int i = 0; i < res.size(); i++)
		cout<<res[i]<<" ";
	cout<<endl;
	int accurateItem = 0;
	for(int i = 0; i < res.size(); i++)
	{
		if(res[i] == CKNN.dataset[i].label)
			accurateItem++;
	}

	cout<<(double)accurateItem / (double)res.size()<<endl;
	long long SUM = countDistComp + countDistComp2 + countDistComp3 + countDistComp4 + countDistComp5;
	cout<<"dist:"<<countDistComp<<" "<<countDistComp / (double)SUM <<endl;
	cout<<"dist2:"<<countDistComp2<<" "<<countDistComp2 / (double)SUM <<endl;
	cout<<"dist3:"<<countDistComp3<<" "<<countDistComp3 / (double)SUM <<endl;
	cout<<"dist4:"<<countDistComp4<<" "<<countDistComp4 / (double)SUM <<endl;
	cout<<"dist5:"<<countDistComp5<<" "<<countDistComp5 / (double)SUM <<endl;
	cout<<"total:"<<SUM<<endl;
	cout<<"grp amount:"<<CKNN.numGroups<<endl;
	cout<<"avg dist:"<<distSum / distCount<<endl;
	return 0;
}
