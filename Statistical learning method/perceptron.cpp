#include <iostream>
#include <vector>
using namespace std;
/*
感知机，在训练数据线性可分的条件下的二分类器，
分类{+1，-1},这里实现感知机的原始形式
*/
class Perceptron{
public:
	typedef vector<vector<int> > matrix;
	typedef vector<int> hvec;
	hvec alphaPrams;
	matrix gram;
	int ata = 1;
	int n,m;
	int dot(hvec &a,hvec &b){
		int ret = 0;
		for(int i = 0;i < a.size();i++)
			ret += a[i]*b[i];
		return ret;
	}
	int __classify(hvec &X,int first,int last){
		int ret = 0;
		for(int i = first;i < last;i++){
			ret += alphaPrams[i]*X[i];
		}
		return ret;
	}
	int classify(hvec &X,int first,int last){
		return __classify(X,first,last) > 0 ? 1 : -1;
	}
	void train(matrix &X){//X最后一列为Y的分类，倒数第二列表示常数项的X，全为1
		n = X.size(),m = X[0].size();
		for(int i = 0;i < m-1;i++) alphaPrams.push_back(0);
		while(1){
			bool allok = true;
			for(int i = 0;i < n;i++){
				int y = X[i][m-1];
				if(y*__classify(X[i],0,m-1) <= 0){
					for(int j = 0;j < m-1;j++) alphaPrams[j] += ata*y*X[i][j];
					allok = false;
					printf("stop at x%d.\n", i);
					break;
				}
			}
			if(allok) break;
		}
	}
	void print_alphaPrams(){
		for(int i = 0;i < alphaPrams.size();i++) printf("%d ", alphaPrams[i]);
		printf("\n");
	}
};
int main(){
	int n,m,x;
	vector<vector<int> > v;
	while(cin >> n >> m){
		m++;//增加一列常数项
		for(int i = 0;i < n;i++){
			vector<int> tv(m,1);//增加常数项1
			for(int j = 0;j < m-2;j++) cin >> tv[j];
			cin >> tv[m-1];
			for(int j = 0;j < m;j++) printf("%d ", tv[j]);
			printf("\n");
			v.push_back(tv);
		}
		Perceptron tron;
		tron.train(v);
		tron.print_alphaPrams();
	}
	return 0;
}
/**数据格式
3 3
3 3 1
4 3 1
1 1 -1
*/