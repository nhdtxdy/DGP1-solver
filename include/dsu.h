#ifndef DSU_H
#define DSU_H

#include <vector>

class DSU {
public:
	DSU() {}
	DSU(int n);
	void unite(int x, int y);
	int num_ccs() const;
	bool same(int x, int y);
	int find(int x);
	int getsz(int x);

private:
	int ccs;
	std::vector<int> par;
	std::vector<int> sz;
};

#endif // DSU_H