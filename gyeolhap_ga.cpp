// =====================================================================
//  Gyeolhap game - GA that designs the PAIR-PICKING ORDER
// ---------------------------------------------------------------------
//  Output : the exact order of card-index pairs to check, printed as
//           a ready-to-paste line for OptAl:
//             vector<pair<int,int>> order = {{1,2},{4,6},...};
//
//  Model
//   - 9 cards on the board, indices 0..8. A pair (i,j) = one check:
//     compute the third card (3 mods) and look it up in HasCard.
//     cost = 3*t_mod + t_y (hit) or 3*t_mod + t_n (miss)
//   - 36 pairs = edges of K9, 84 triples = triangles of K9.
//     Checking pair (i,j) decides the 7 triples {i,j,k}.
//   - Chromosome = permutation of the 36 pairs. Decoding keeps a pair
//     only if it decides at least one new triple, and stops once all
//     84 triples are decided ("no hap" is then proven).
//     Mantel: min number of kept pairs = 36 - floor(81/4) = 16.
//
//  Fitness (lower is better), measured on a fixed sample of boards:
//     E[cost until first hap (or proof of none)]
//   + W_COUNT * E[cost to count all haps with the same plan]
//
//  GA : tournament selection, elitism, OX crossover,
//       swap/insert/reverse mutation (biased to the front),
//       greedy seeds, local search when the best stops improving.
//
//  Build (Dev-C++): Tools > Compiler Options > add  -std=c++11 -O2
//  Command line   : g++ -std=c++11 -O2 -o ga_order gyeolhap_ga_order.cpp
//  Run            : ga_order [T=1000000] [W_COUNT=0.05] [seed]
//  Pure ASCII, single-threaded, no __int128.
// =====================================================================
#define __USE_MINGW_ANSI_STDIO 1
#if __cplusplus < 201103L
#error "Compile with -std=c++11 (Dev-C++: Tools > Compiler Options)"
#endif
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64;

// ---------------- user cost model ----------------
const double t_mod = 618.22, t_y = 1717.71, t_n = 2174.28;
const double C_HIT  = 3 * t_mod + t_y;
const double C_MISS = 3 * t_mod + t_n;

const int NC = 9, NE = 36, NT = 84;

// ---------------- K9 geometry ----------------
int EU[NE], EV[NE], edgeId[NC][NC];
int TA[NT], TB[NT], TC[NT];
u64 EMlo[NE], EMhi[NE];          // triangles decided by edge e (84 bits = lo 64 + hi 20)

void initGeometry() {
    int e = 0;
    for (int i = 0; i < NC; i++)
        for (int j = i + 1; j < NC; j++) { EU[e] = i; EV[e] = j; edgeId[i][j] = edgeId[j][i] = e; e++; }
    int t = 0;
    for (int i = 0; i < NC; i++)
        for (int j = i + 1; j < NC; j++)
            for (int k = j + 1; k < NC; k++) {
                TA[t] = i; TB[t] = j; TC[t] = k;
                int es[3] = { edgeId[i][j], edgeId[i][k], edgeId[j][k] };
                for (int q = 0; q < 3; q++) {
                    if (t < 64) EMlo[es[q]] |= 1ULL << t;
                    else        EMhi[es[q]] |= 1ULL << (t - 64);
                }
                t++;
            }
}
const u64 ALL_LO = ~0ULL, ALL_HI = (1ULL << (NT - 64)) - 1;
inline int pc(u64 v) { return __builtin_popcountll(v); }

// ---------------- cards (same encoding as the user's harness) ----------------
// card id 0..26  ->  a = id/9, b = (id/3)%3, c = id%3
inline int thirdId(int p, int q) {
    int a = (6 - p / 9 - q / 9) % 3;
    int b = (6 - (p / 3) % 3 - (q / 3) % 3) % 3;
    int c = (6 - p % 3 - q % 3) % 3;
    return a * 9 + b * 3 + c;
}
// For one board: bit e is set if checking pair e finds a hap.
u64 boardHitMask(const int* id) {
    bool has[27] = {};
    for (int i = 0; i < NC; i++) has[id[i]] = true;
    u64 m = 0;
    for (int e = 0; e < NE; e++) if (has[thirdId(id[EU[e]], id[EV[e]])]) m |= 1ULL << e;
    return m;
}

// ---------------- chromosome -> plan ----------------
typedef array<int, NE> Perm;
struct Plan { int len; int e[NE]; u64 mask; };

Plan decode(const Perm& p) {
    Plan P; P.len = 0; P.mask = 0;
    u64 lo = 0, hi = 0;
    for (int k = 0; k < NE; k++) {
        if (lo == ALL_LO && hi == ALL_HI) break;               // all triples decided
        u64 nlo = EMlo[p[k]] & ~lo, nhi = EMhi[p[k]] & ~hi;
        if (!nlo && !nhi) continue;                             // nothing new -> skip
        P.e[P.len++] = p[k]; P.mask |= 1ULL << p[k];
        lo |= nlo; hi |= nhi;
    }
    return P;
}

// ---------------- fitness ----------------
double W_COUNT = 0.05;
struct Score { double first, all, fit; int len; };

Score evaluate(const Plan& P, const vector<u64>& S) {
    double sf = 0, sa = 0;
    const double missAll = P.len * C_MISS;
    for (size_t b = 0; b < S.size(); b++) {
        u64 h = S[b];
        int k = 0;
        while (k < P.len && !((h >> P.e[k]) & 1)) k++;
        sf += (k < P.len) ? k * C_MISS + C_HIT : missAll;
        sa += missAll - pc(h & P.mask) * (C_MISS - C_HIT);
    }
    Score r;
    r.first = sf / S.size(); r.all = sa / S.size(); r.len = P.len;
    r.fit = r.first + W_COUNT * r.all;
    return r;
}

// ---------------- GA operators ----------------
struct Ind { Perm p; Score s; };
bool better(const Ind& a, const Ind& b) { return a.s.fit < b.s.fit; }

Perm randomPerm(mt19937& g) { Perm p; iota(p.begin(), p.end(), 0); shuffle(p.begin(), p.end(), g); return p; }

// greedy: each step take the pair deciding the most new triples (ties random)
Perm greedyPerm(mt19937& g) {
    Perm p; vector<int> rest(NE); iota(rest.begin(), rest.end(), 0);
    u64 lo = 0, hi = 0;
    for (int k = 0; k < NE; k++) {
        int best = -1; vector<int> cand;
        for (int i = 0; i < (int)rest.size(); i++) {
            int gain = pc(EMlo[rest[i]] & ~lo) + pc(EMhi[rest[i]] & ~hi);
            if (gain > best) { best = gain; cand.clear(); }
            if (gain == best) cand.push_back(i);
        }
        int pick = cand[g() % cand.size()];
        p[k] = rest[pick]; lo |= EMlo[p[k]]; hi |= EMhi[p[k]];
        rest.erase(rest.begin() + pick);
    }
    return p;
}

Perm orderCrossover(const Perm& a, const Perm& b, mt19937& g) {
    int l = g() % NE, r = g() % NE; if (l > r) swap(l, r);
    Perm c; bool used[NE] = {};
    for (int i = l; i <= r; i++) { c[i] = a[i]; used[a[i]] = true; }
    int pos = (r + 1) % NE;
    for (int k = 0; k < NE; k++) {
        int v = b[(r + 1 + k) % NE];
        if (!used[v]) { c[pos] = v; pos = (pos + 1) % NE; }
    }
    return c;
}

int frontPos(mt19937& g) { return (g() % 4) ? g() % 22 : g() % NE; }
void moveItem(Perm& p, int i, int j) {             // remove p[i], insert at j
    int v = p[i];
    if (i < j) for (int k = i; k < j; k++) p[k] = p[k + 1];
    else       for (int k = i; k > j; k--) p[k] = p[k - 1];
    p[j] = v;
}
void mutate(Perm& p, mt19937& g) {
    int i = frontPos(g), j = frontPos(g);
    if (i == j) j = (j + 1) % NE;
    switch (g() % 3) {
        case 0: swap(p[i], p[j]); break;
        case 1: moveItem(p, i, j); break;
        default: if (i > j) swap(i, j); reverse(p.begin() + i, p.begin() + j + 1);
    }
}
// local search: try every "move one pair to another slot" near the front
bool localSearch(Ind& best, const vector<u64>& S) {
    bool improved = false, again = true;
    while (again) {
        again = false;
        int lim = min(NE, decode(best.p).len + 6);
        for (int i = 0; i < NE && !again; i++)
            for (int j = 0; j < lim && !again; j++) {
                if (i == j) continue;
                Ind c = best; moveItem(c.p, i, j);
                c.s = evaluate(decode(c.p), S);
                if (c.s.fit < best.s.fit - 1e-9) { best = c; again = improved = true; }
            }
    }
    return improved;
}

// ---------------- relabel cards so group A = 0..3, group B = 4..8 ----------------
// Any relabeling is equally good (the 9 cards are random), this only makes the
// output match the user's OptAl convention.
vector<pair<int,int> > canonicalOrder(const Plan& P) {
    // connected components of the checked-pair graph
    int comp[NC]; fill(comp, comp + NC, -1); int nc = 0;
    for (int s = 0; s < NC; s++) if (comp[s] < 0) {
        vector<int> st(1, s); comp[s] = nc;
        while (!st.empty()) {
            int u = st.back(); st.pop_back();
            for (int k = 0; k < P.len; k++) {
                int a = EU[P.e[k]], b = EV[P.e[k]], v = -1;
                if (a == u) v = b; else if (b == u) v = a;
                if (v >= 0 && comp[v] < 0) { comp[v] = nc; st.push_back(v); }
            }
        }
        nc++;
    }
    vector<int> sz(nc, 0); for (int v = 0; v < NC; v++) sz[comp[v]]++;
    // smaller component first, inside a component by first appearance in the plan
    vector<int> rank(NC, 1000), label(NC, -1);
    for (int k = P.len - 1; k >= 0; k--) { rank[EU[P.e[k]]] = k * 2; rank[EV[P.e[k]]] = k * 2 + 1; }
    vector<int> vs(NC); iota(vs.begin(), vs.end(), 0);
    sort(vs.begin(), vs.end(), [&](int a, int b) {
        if (sz[comp[a]] != sz[comp[b]]) return sz[comp[a]] < sz[comp[b]];
        if (comp[a] != comp[b]) return comp[a] < comp[b];
        return rank[a] < rank[b];
    });
    for (int i = 0; i < NC; i++) label[vs[i]] = i;
    vector<pair<int,int> > ord;
    for (int k = 0; k < P.len; k++) {
        int a = label[EU[P.e[k]]], b = label[EV[P.e[k]]];
        if (a > b) swap(a, b);
        ord.push_back(make_pair(a, b));
    }
    return ord;
}

// ---------------- validation helpers (user's harness style) ----------------
struct S3 { int a, b, c; };
int HasCard[3][3][3];
pair<int,double> runOrder(const vector<S3>& cards, const vector<pair<int,int> >& order) {
    double steps = 0;
    for (size_t i = 0; i < order.size(); i++) {
        const S3& p = cards[order[i].first]; const S3& q = cards[order[i].second];
        int a = (6 - p.a - q.a) % 3, b = (6 - p.b - q.b) % 3, c = (6 - p.c - q.c) % 3;
        steps += 3 * t_mod;
        if (HasCard[a][b][c]) { steps += t_y; return make_pair(0, steps); }
        steps += t_n;
    }
    return make_pair(1, steps);
}
int bruteCount(const vector<S3>& cards) {
    int h = 0;
    for (int i = 0; i < NC; i++) for (int j = i + 1; j < NC; j++) {
        int a = (6 - cards[i].a - cards[j].a) % 3, b = (6 - cards[i].b - cards[j].b) % 3, c = (6 - cards[i].c - cards[j].c) % 3;
        if (HasCard[a][b][c]) h++;
    }
    return h / 3;
}

// =====================================================================
int main(int argc, char** argv) {
    long long T = argc > 1 ? atoll(argv[1]) : 1000000;
    if (argc > 2) W_COUNT = atof(argv[2]);
    unsigned seed = argc > 3 ? (unsigned)atoll(argv[3])
                             : (unsigned)chrono::high_resolution_clock::now().time_since_epoch().count();
    initGeometry();
    mt19937 g(seed);
    vector<int> deck(27); iota(deck.begin(), deck.end(), 0);

    // ---- training sample: same boards for every individual (noise-free comparison) ----
    const int SAMPLE = 200000;
    vector<u64> S(SAMPLE);
    for (int b = 0; b < SAMPLE; b++) { shuffle(deck.begin(), deck.end(), g); S[b] = boardHitMask(deck.data()); }

    // ---- GA parameters ----
    const int POP = 200, ELITE = 4, TOUR = 3, MAXGEN = 800, PATIENCE = 80;
    const int PC = 900, PM = 350;   // per mille

    vector<Ind> pop(POP);
    for (int i = 0; i < POP; i++) {
        pop[i].p = (i < POP / 5) ? greedyPerm(g) : randomPerm(g);
        pop[i].s = evaluate(decode(pop[i].p), S);
    }
    sort(pop.begin(), pop.end(), better);
    Ind best = pop[0];

    printf("[GA] seed=%u  sample=%d boards  pop=%d  W_COUNT=%.3f\n", seed, SAMPLE, POP, W_COUNT);
    printf("%5s %12s %12s %12s %4s\n", "gen", "fitness", "E[first]", "E[all]", "len");

    int stall = 0, gen;
    for (gen = 0; gen < MAXGEN && stall < PATIENCE; gen++) {
        vector<Ind> nxt(pop.begin(), pop.begin() + ELITE);
        while ((int)nxt.size() < POP) {
            int a = g() % POP, b = g() % POP;
            for (int k = 1; k < TOUR; k++) { int c = g() % POP; if (pop[c].s.fit < pop[a].s.fit) a = c; }
            for (int k = 1; k < TOUR; k++) { int c = g() % POP; if (pop[c].s.fit < pop[b].s.fit) b = c; }
            Ind ch;
            ch.p = ((int)(g() % 1000) < PC) ? orderCrossover(pop[a].p, pop[b].p, g) : pop[a].p;
            if ((int)(g() % 1000) < PM) mutate(ch.p, g);
            if (g() % 10 == 0) mutate(ch.p, g);
            ch.s = evaluate(decode(ch.p), S);
            nxt.push_back(ch);
        }
        pop.swap(nxt);
        sort(pop.begin(), pop.end(), better);

        if (pop[0].s.fit < best.s.fit - 1e-9) { best = pop[0]; stall = 0; } else stall++;
        if (stall > 0 && stall % 20 == 0 && localSearch(best, S)) {
            pop[POP - 1] = best; sort(pop.begin(), pop.end(), better); stall = 0;
        }
        if (gen % 10 == 0 || stall == 0)
            printf("%5d %12.2f %12.2f %12.2f %4d\n", gen, best.s.fit, best.s.first, best.s.all, best.s.len);
    }
    localSearch(best, S);
    Plan P = decode(best.p);
    vector<pair<int,int> > order = canonicalOrder(P);
    printf("[Done] gen %d   E[first]=%.2f   E[all]=%.2f   pairs=%d\n", gen, best.s.first, best.s.all, P.len);

    // ---- step-by-step picking order ----
    // relabeled plan for the table
    Plan Q; Q.len = 0; Q.mask = 0;
    for (size_t k = 0; k < order.size(); k++) { int e = edgeId[order[k].first][order[k].second]; Q.e[Q.len++] = e; Q.mask |= 1ULL << e; }
    printf("\n===== Picking order (card indices 0..8, group A = 0-3, group B = 4-8) =====\n");
    printf("%4s %7s %9s %10s %12s %12s\n", "step", "pair", "new trip", "decided", "P(hit here)", "P(found)");
    {
        // re-sample relabeled hit masks so the probabilities match the printed labels
        vector<u64> SV(SAMPLE);
        mt19937 gs(seed + 99);
        for (int b = 0; b < SAMPLE; b++) { shuffle(deck.begin(), deck.end(), gs); SV[b] = boardHitMask(deck.data()); }
        vector<long long> hitAt(Q.len + 1, 0);
        for (int b = 0; b < SAMPLE; b++) {
            int k = 0; while (k < Q.len && !((SV[b] >> Q.e[k]) & 1)) k++;
            hitAt[k]++;
        }
        u64 lo = 0, hi = 0; double cum = 0;
        for (int k = 0; k < Q.len; k++) {
            int nw = pc(EMlo[Q.e[k]] & ~lo) + pc(EMhi[Q.e[k]] & ~hi);
            lo |= EMlo[Q.e[k]]; hi |= EMhi[Q.e[k]];
            double ph = (double)hitAt[k] / SAMPLE; cum += ph;
            printf("%4d   (%d,%d) %9d %7d/84 %11.4f%% %11.4f%%\n",
                   k + 1, order[k].first, order[k].second, nw, pc(lo) + pc(hi), 100 * ph, 100 * cum);
        }
        printf("  no hap on board: %.4f%%  (proven after all %d pairs)\n", 100.0 * hitAt[Q.len] / SAMPLE, Q.len);
    }

    // ---- paste-ready line ----
    string line = "vector<pair<int,int>> order = {";
    for (size_t k = 0; k < order.size(); k++) {
        char buf[32]; sprintf(buf, "{%d, %d}%s", order[k].first, order[k].second, k + 1 < order.size() ? ", " : "};");
        line += buf;
    }
    printf("\n===== Paste into OptAl =====\n%s\n", line.c_str());
    FILE* f = fopen("ga_order.txt", "w");
    if (f) { fprintf(f, "// GA picking order (seed=%u)\n%s\n", seed, line.c_str()); fclose(f); }

    // ---- validation on fresh boards, exactly like the user's harness ----
    vector<pair<int,int> > lex;     // user's lexicographic K4+K5 order, for comparison
    for (int i = 0; i < 4; i++) for (int j = i + 1; j < 4; j++) lex.push_back(make_pair(i, j));
    for (int i = 4; i < 9; i++) for (int j = i + 1; j < 9; j++) lex.push_back(make_pair(i, j));
    vector<pair<int,int> > brute;
    for (int i = 0; i < 9; i++) for (int j = i + 1; j < 9; j++) brute.push_back(make_pair(i, j));

    mt19937 gv(seed ^ 0x9E3779B9u);
    vector<S3> cards(NC);
    double cost[3] = {0, 0, 0}; long long fails[3] = {0, 0, 0}, wrong[3] = {0, 0, 0};
    const vector<pair<int,int> >* plans[3] = { &brute, &lex, &order };
    for (long long it = 0; it < T; it++) {
        memset(HasCard, 0, sizeof(HasCard));
        shuffle(deck.begin(), deck.end(), gv);
        for (int j = 0; j < NC; j++) {
            int id = deck[j];
            cards[j].a = id / 9; cards[j].b = (id / 3) % 3; cards[j].c = id % 3;
            HasCard[cards[j].a][cards[j].b][cards[j].c] = 1;
        }
        int ref = bruteCount(cards);
        for (int q = 0; q < 3; q++) {
            pair<int,double> r = runOrder(cards, *plans[q]);
            cost[q] += r.second; fails[q] += r.first;
            if ((r.first == 1) != (ref == 0)) wrong[q]++;
        }
    }
    const char* nm[3] = { "Brute-Force (36 pairs)", "K4+K5 lexicographic", "GA order" };
    printf("\n===== Validation on %lld fresh boards =====\n", T);
    printf("%-24s %14s %10s %8s %10s\n", "algorithm", "avg steps", "no-hap", "errors", "vs brute");
    for (int q = 0; q < 3; q++)
        printf("%-24s %14.2f %10lld %8lld %9.2f%%\n", nm[q], cost[q] / T, fails[q], wrong[q], 100.0 * (1 - cost[q] / cost[0]));
    printf("\nSaved: ga_order.txt\n");
    return 0;
}
