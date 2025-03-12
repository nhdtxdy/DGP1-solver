#include "dsu.h"
#include <numeric>
using namespace std;

DSU::DSU(int n) {
	this->ccs = n;
	par.resize(n + 1);
	sz.resize(n + 1, 1);
	iota(par.begin(), par.end(), 0);
}

void DSU::unite(int x, int y) {
	x = find(x);
	y = find(y);
	if (x == y) return;
	--ccs;
	if (sz[x] < sz[y]) swap(x, y);
	par[y] = x;
	sz[x] += sz[y];
}

int DSU::num_ccs() const {
	return ccs;
}

int DSU::find(int x) {
	if (x == par[x]) return x;
	return par[x] = find(par[x]);
}

bool DSU::same(int x, int y) {
	return find(x) == find(y);
}

int DSU::getsz(int x) {
	int par = find(x);
	return sz[par];
}
