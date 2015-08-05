/*
hdu 4347 The Closest M Points
K近邻模型，kd树实现
*/
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <string>
using namespace std;
typedef long long ll;
const int inf = 1<<30;
const ll ll_inf = 1ll<<60;
#define maxN 50005
#define maxD 6
#define	maxK 12
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
int n,m,K;//n是点的个数，m是维度,K是最近邻的个数
struct point{
	int x[maxD];//特征向量
	void read(){
		for(int i = 0;i < m;i++) scanf("%d",&x[i]);
	}
}pnts[maxN];
/*
以[first,last)区间的点构建kd树，按照最大间隔的一维now分割，
并且使得在now维上以mid为分割点进行partition
（在李航的书里是按照方差进行选出当前的一维）
*/
int now,divX[maxN*2];//divX[i]记录点i对应的分割维度
bool cmp(const point &a,const point &b){
	return a.x[now] < b.x[now];
}
void build(int first,int last,point p[]){
	if(first >= last) return;
	int maxx[maxD],minx[maxD];
	for(int i = 0;i < m;i++) maxx[i] = -inf,minx[i] = inf;
	for(int i = first;i < last;i++)
		for(int j = 0;j < m;j++){
			maxx[j] = MAX(maxx[j],p[i].x[j]);
			minx[j] = MIN(minx[j],p[i].x[j]);
		}
	now = 0;
	for(int i = 1;i < m;i++)
		if(maxx[i]-minx[i] > maxx[now]-minx[now]) now = i;
	int mid = first+last>>1;
	divX[mid] = now;
	nth_element(p+first,p+mid,p+last,cmp);
	if(first+1 == last) return;// 如果只有一个元素，不需要再向下递归
	build(first,mid,p);
	build(mid+1,last,p);
}
ll closestDist[maxK];
int closestNodes[maxK][maxD];
//更新K近邻,找到最先能够更新的位置i，然后把原来的[i,m)集体后移，最后再更新i位置
void update(ll d,int px[]){
	for(int i = 0;i < K;i++)
		if(closestDist[i] > d){
			for(int j = K-1;j > i;j--){
				closestDist[j] = closestDist[j-1];
				for(int k = 0;k < m;k++) 
					closestNodes[j][k] = closestNodes[j-1][k];
			}
			for(int k = 0;k < m;k++) closestNodes[i][k] = px[k];
			closestDist[i] = d;
			return; 
		}
}
void findK(int first,int last,const point &p){
	if(first >= last) return;
	int mid = first+last>>1;
	ll d = 0;
	for(int i = 0;i < m;i++){
		ll tmp = pnts[mid].x[i]-p.x[i];
		d += tmp*tmp;
	}
	update(d,pnts[mid].x);
	if(first + 1 == last) return;//只有一个点，不用再继续找下去
	int divx = divX[mid];
	ll delta = p.x[divx]-pnts[mid].x[divx];
	ll delta2 = delta*delta;
	int l1 = first,r1 = mid;
	int l2 = mid+1,r2 = last;
	if(delta > 0){// 点p的divx维小于mid的divx维,继续左半部分，否则右半部分
		swap(l1,l2);
		swap(r1,r2);
	}
	findK(l1,r1,p);
	if(delta2 < closestDist[K-1])//如果回溯时发现点p到mid的divx维的距离比K近邻中最大的一个小，搜索另一部分
		findK(l2,r2,p);
}
void findNearestNeighbour(const point &p){
	for(int i = 0;i < K;i++)
		closestDist[i] = ll_inf;
	findK(0,n,p);
}
void print(){
	printf("the closest %d points are:\n",K);
	for(int i = 0;i < K;i++){
		for(int j = 0;j < m-1;j++) printf("%d ", closestNodes[i][j]);
		printf("%d\n", closestNodes[i][m-1]);
	}
}
int main(){
	point P;
	while(~scanf("%d %d",&n,&m)){
		for(int i = 0;i < n;i++) pnts[i].read();
		build(0,n,pnts);
		int q;
		scanf("%d",&q);
		while(q--){
			P.read();
			scanf("%d",&K);
			findNearestNeighbour(P);
			print();
		}
	}
	return 0;
}