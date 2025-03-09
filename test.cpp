#include <bits/stdc++.h>
using namespace std;

void merge(vector<unordered_map<int, double>> &base, vector<unordered_map<int, double>> to_merge) {
    vector<unordered_map<int, double>> merged;
    for (const unordered_map<int, double>& b : base) {
        for (const unordered_map<int, double> &tm : to_merge) {
            unordered_map<int, double> cpy = b;
            cpy.insert(tm.begin(), tm.end());
            merged.push_back(cpy);
        }
    }
    swap(base, merged);
}

int main() {
	vector<unordered_map<int, double>> test, tm;

	unordered_map<int, double> t1;
	t1[0] = 1;

	unordered_map<int, double> tm1;
	tm1[1] = 2;
	tm1[2] = 3;

	unordered_map<int, double> tm2;
	tm2[3] = 4;
	tm2[4] = 5;

	test.push_back(t1);
	tm.push_back(tm1);
	tm.push_back(tm2);

	merge(test, tm);

	cerr << test.size() << '\n';

	for (int i = 0; i < test.size(); ++i)
	for (auto &p : test[i]) {
		cerr << p.first << ' ' << p.second << '\n';
	}
}