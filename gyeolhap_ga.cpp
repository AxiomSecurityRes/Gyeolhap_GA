// =====================================================================
//  Gyeolhap (SET-like) game - Genetic Algorithm optimizer for the pair-check order (plan)
// ---------------------------------------------------------------------
//  [Problem model]
//   - 27 cards = 3 attributes x 3 values (points of AG(3,3)), 9 cards on the board.
//   - A 'hap' = 3 cards whose every attribute sums to 0 (mod 3).
//   - Basic op: pick a pair (i,j), compute the third card (3 mods), look up HasCard.
//     Cost model is the user's measurement: 3*t_mod + (t_y | t_n).
//
//  [Key idea]
//   Position pairs of 9 cards = 36 edges of K9, possible triples = 84 triangles.
//   Checking pair (i,j) decides the 7 triangles {i,j,k} (they become 'covered').
//   -> One edge permutation IS an algorithm. While decoding:
//      (1) skip edges that cover no new triangle (decided offline, zero runtime cost)
//      (2) once all 84 triangles are covered, conclude 'no hap' and stop.
//   Mantel's theorem: max edges of a triangle-free graph on 9 vertices = floor(81/4) = 20
//   -> minimum checks needed to cover every triangle = 36 - 20 = 16 (K4 U K5).
//   The GA searches WHICH 16 pairs and in WHAT ORDER, minimizing expected cost.
//
//  [GA setup]
//   Chromosome : permutation of the 36 edges
//   Evaluation : expected cost on a fixed board sample (common random numbers, CRN)
//                fitness = E[cost to first hap] + W_COUNT * E[cost to count all haps]
//   Selection  : tournament (k=3) + elitism
//   Crossover  : OX (Order Crossover)
//   Mutation   : swap / insertion / inversion (biased toward the front)
//   Speed-up   : greedy seeds + elite local search on stagnation (memetic)
//
//  [Validation] T fresh boards compared 1:1 with BruteForce (ground truth) + real timing.
//  Build: g++ -std=c++11 -O2 -o ga gyeolhap_ga.cpp   (optional: -fopenmp for multicore)
//  Run  : ./ga [T=1000000] [W_COUNT=0.05] [seed]
// =====================================================================
#define __USE_MINGW_ANSI_STDIO 1   // fix %lld etc. in printf on old MinGW
#if __cplusplus < 201103L
#error "Compile with C++11 or later. Dev-C++: Tools > Compiler Options > add -std=c++11 to compiler commands"
#endif
#include <bits/stdc++.h>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
typedef unsigned __int128 u128;

// ---------------- user cost model ----------------
const double t_mod = 618.22, t_y = 1717.71, t_n = 2174.28;
const double C_HIT  = 3 * t_mod + t_y;
const double C_MISS = 3 * t_mod + t_n;

const int NC = 9, NE = 36, NT = 84;

// ---------------- geometry (K9 edges / triangles) ----------------
int EU[NE], EV[NE], edgeId[NC][NC];
int TA[NT], TB[NT], TC[NT], triId[NC][NC][NC];
u128 EM[NE];      // set of triangles covered by edge e
u128 ALL_TRI;

void initGeometry() {
    int e = 0;
    for (int i = 0; i < NC; i++)
        for (int j = i + 1; j < NC; j++) { EU[e] = i; EV[e] = j; edgeId[i][j] = edgeId[j][i] = e; e++; }
    int t = 0;
    for (int i = 0; i < NC; i++)
        for (int j = i + 1; j < NC; j++)
            for (int k = j + 1; k < NC; k++) {
                TA[t] = i; TB[t] = j; TC[t] = k;
                int p[3] = {i, j, k};
                sort(p, p + 3);
                do triId[p[0]][p[1]][p[2]] = t; while (next_permutation(p, p + 3));
                u128 bit = (u128)1 << t;
                EM[edgeId[i][j]] |= bit; EM[edgeId[i][k]] |= bit; EM[edgeId[j][k]] |= bit;
                t++;
            }
    ALL_TRI = ((u128)1 << NT) - 1;
}

// ---------------- board generation (same distribution as user code) ----------------
// user code: shuffle 1..27, take first 9, BaseConv(idx,3,3) = (idx-1) in base 3, 3 digits
// -> card id = idx-1, a = id/9, b = (id/3)%3, c = id%3
struct Board {
    uint8_t id[NC];
    u128 hap;          // set of triangles that are haps
};
inline int A_(int id) { return id / 9; }
inline int B_(int id) { return (id / 3) % 3; }
inline int C_(int id) { return id % 3; }
inline int thirdId(int p, int q) {
    int a = (6 - A_(p) - A_(q)) % 3, b = (6 - B_(p) - B_(q)) % 3, c = (6 - C_(p) - C_(q)) % 3;
    return a * 9 + b * 3 + c;
}

Board makeBoard(mt19937& g, array<int, 27>& deck) {
    shuffle(deck.begin(), deck.end(), g);
    Board B;
    for (int i = 0; i < NC; i++) B.id[i] = deck[i];
    B.hap = 0;
    for (int t = 0; t < NT; t++)
        if (thirdId(B.id[TA[t]], B.id[TB[t]]) == B.id[TC[t]]) B.hap |= (u128)1 << t;
    return B;
}

// ---------------- chromosome -> plan decoding ----------------
typedef array<uint8_t, NE> Perm;
struct Plan {
    int len = 0;
    uint8_t e[NE];
    u128 newCov[NE];   // triangles first covered at this step (dedup for counting)
};
Plan decode(const Perm& p) {
    Plan P;
    u128 cov = 0;
    for (int k = 0; k < NE && cov != ALL_TRI; k++) {
        u128 add = EM[p[k]] & ~cov;
        if (!add) continue;                // no new info -> skip
        P.e[P.len] = p[k];
        P.newCov[P.len] = add;
        P.len++;
        cov |= add;
    }
    return P;  // cov == ALL_TRI guaranteed (union of all edges = all)
}

// ---------------- evaluation function ----------------
double W_COUNT = 0.05;
struct Score { double first, count, fit; int len; };

Score evaluate(const Plan& P, const vector<Board>& S) {
    double sf = 0, sc = 0;
    for (const Board& b : S) {
        double c = 0, call = 0;
        bool hit = false;
        for (int k = 0; k < P.len; k++) {
            bool h = (b.hap & EM[P.e[k]]) != 0;
            double ck = h ? C_HIT : C_MISS;
            call += ck;
            if (!hit) { c += ck; if (h) hit = true; }
        }
        sf += c; sc += call;
    }
    Score r;
    r.first = sf / S.size();
    r.count = sc / S.size();
    r.len = P.len;
    r.fit = r.first + W_COUNT * r.count;
    return r;
}

// ---------------- GA operators ----------------
struct Ind { Perm p; Score s; };

Perm randomPerm(mt19937& g) {
    Perm p;
    iota(p.begin(), p.end(), 0);
    shuffle(p.begin(), p.end(), g);
    return p;
}
// greedy seed: each step pick the edge covering the most new triangles (ties random)
Perm greedyPerm(mt19937& g) {
    Perm p; vector<int> rest(NE); iota(rest.begin(), rest.end(), 0);
    u128 cov = 0;
    for (int k = 0; k < NE; k++) {
        int best = -1; vector<int> cand;
        for (int i = 0; i < (int)rest.size(); i++) {
            int gain = __builtin_popcountll((uint64_t)(EM[rest[i]] & ~cov)) +
                       __builtin_popcountll((uint64_t)((EM[rest[i]] & ~cov) >> 64));
            if (gain > best) { best = gain; cand.clear(); }
            if (gain == best) cand.push_back(i);
        }
        int pick = cand[g() % cand.size()];
        p[k] = rest[pick];
        cov |= EM[rest[pick]];
        rest.erase(rest.begin() + pick);
    }
    return p;
}
// OX crossover
Perm orderCrossover(const Perm& a, const Perm& b, mt19937& g) {
    int l = g() % NE, r = g() % NE;
    if (l > r) swap(l, r);
    Perm c; bool used[NE] = {};
    for (int i = l; i <= r; i++) { c[i] = a[i]; used[a[i]] = true; }
    int pos = (r + 1) % NE;
    for (int k = 0; k < NE; k++) {
        int v = b[(r + 1 + k) % NE];
        if (used[v]) continue;
        c[pos] = v; pos = (pos + 1) % NE;
    }
    return c;
}
// mutation: the plan is decided by the first ~20 slots, so bias positions to the front
int biasedPos(mt19937& g) { return (g() % 4) ? g() % 22 : g() % NE; }
void mutate(Perm& p, mt19937& g) {
    int op = g() % 3, i = biasedPos(g), j = biasedPos(g);
    if (i == j) j = (j + 1) % NE;
    if (op == 0) swap(p[i], p[j]);
    else if (op == 1) {                         // insertion
        uint8_t v = p[i];
        if (i < j) for (int k = i; k < j; k++) p[k] = p[k + 1];
        else       for (int k = i; k > j; k--) p[k] = p[k - 1];
        p[j] = v;
    } else { if (i > j) swap(i, j); reverse(p.begin() + i, p.begin() + j + 1); }
}
// local search (memetic): scan all insertion neighbors of the front, first-improvement
bool localSearch(Ind& best, const vector<Board>& S) {
    bool improved = false, again = true;
    while (again) {
        again = false;
        int lim = min(NE, decode(best.p).len + 6);
        vector<pair<int,int>> moves;
        for (int i = 0; i < NE; i++) for (int j = 0; j < lim; j++) if (i != j) moves.push_back({i, j});
        vector<Ind> cand(moves.size());
        #pragma omp parallel for schedule(dynamic, 16)
        for (int m = 0; m < (int)moves.size(); m++) {
            Perm q = best.p; int i = moves[m].first, j = moves[m].second;
            uint8_t v = q[i];
            if (i < j) for (int k = i; k < j; k++) q[k] = q[k + 1];
            else       for (int k = i; k > j; k--) q[k] = q[k - 1];
            q[j] = v;
            cand[m].p = q; cand[m].s = evaluate(decode(q), S);
        }
        for (auto& c : cand)
            if (c.s.fit < best.s.fit - 1e-9) { best = c; again = improved = true; }
    }
    return improved;
}

// ---------------- baseline (same order/cost as user's BruteAl) ----------------
// returns: {failed, cost}
pair<int, double> bruteAl(const uint8_t* id, const bool* has) {
    double steps = 0;
    for (int i = 0; i < NC; i++)
        for (int j = i + 1; j < NC; j++) {
            steps += 3 * t_mod;
            if (has[thirdId(id[i], id[j])]) { steps += t_y; return {0, steps}; }
            steps += t_n;
        }
    return {1, steps};
}
int bruteForceCount(const uint8_t* id, const bool* has) {
    int hap = 0;
    for (int i = 0; i < NC; i++) for (int j = i + 1; j < NC; j++) if (has[thirdId(id[i], id[j])]) hap++;
    return hap / 3;
}

// =====================================================================
int main(int argc, char** argv) {
    long long T = argc > 1 ? atoll(argv[1]) : 1000000;
    if (argc > 2) W_COUNT = atof(argv[2]);
    unsigned seed = argc > 3 ? (unsigned)atoll(argv[3])
                             : (unsigned)chrono::high_resolution_clock::now().time_since_epoch().count();
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);   // UTF-8 console output
#endif
    initGeometry();
    mt19937 g(seed);
    array<int, 27> deck; iota(deck.begin(), deck.end(), 0);

    // ---- training sample (CRN: every individual sees the same boards -> noise-free comparison) ----
    const int SAMPLE = 120000;
    vector<Board> S(SAMPLE);
    for (auto& b : S) b = makeBoard(g, deck);

    // ---- initial population ----
    const int POP = 160, ELITE = 4, TOUR = 3, MAXGEN = 600, PATIENCE = 60;
    const double PC = 0.9, PM = 0.35;
    vector<Ind> pop(POP);
    for (int i = 0; i < POP; i++) pop[i].p = (i < POP / 5) ? greedyPerm(g) : randomPerm(g);
    #pragma omp parallel for
    for (int i = 0; i < POP; i++) pop[i].s = evaluate(decode(pop[i].p), S);

    auto byFit = [](const Ind& a, const Ind& b) { return a.s.fit < b.s.fit; };
    sort(pop.begin(), pop.end(), byFit);
    Ind best = pop[0];
    // keep the initial best (greedy seed) plan to measure the GA gain honestly
    Ind greedyBest = pop[0];
    for (auto& q : pop) if (q.s.fit < greedyBest.s.fit) greedyBest = q;
    Plan PG = decode(greedyBest.p);

    // baseline (BruteAl order = lexicographic 0..35, all 36 checks, no skip / early stop)
    double baseFirst = 0, baseCount = 0;
    for (const Board& b : S) {
        double c = 0; bool hit = false;
        for (int e = 0; e < NE; e++) {
            bool h = (b.hap & EM[e]) != 0;
            double ck = h ? C_HIT : C_MISS;
            baseCount += ck;
            if (!hit) { c += ck; if (h) hit = true; }
        }
        baseFirst += c;
    }
    baseFirst /= SAMPLE; baseCount /= SAMPLE;

    printf("[GA] seed=%u  sample=%d boards  W_COUNT=%.3f\n", seed, SAMPLE, W_COUNT);
    printf("[Base] BruteAl order: first hap %.1f / count all %.1f (36 checks)\n", baseFirst, baseCount);
    printf("%5s %12s %12s %12s %5s\n", "gen", "fitness", "E[first]", "E[all]", "len");

    int stall = 0, gen;
    for (gen = 0; gen < MAXGEN && stall < PATIENCE; gen++) {
        vector<Ind> nxt(pop.begin(), pop.begin() + ELITE);
        vector<mt19937> rngs;
        nxt.resize(POP);
        auto tour = [&](mt19937& r) -> const Ind& {
            int b = r() % POP;
            for (int k = 1; k < TOUR; k++) { int c = r() % POP; if (pop[c].s.fit < pop[b].s.fit) b = c; }
            return pop[b];
        };
        #pragma omp parallel
        {
            #ifdef _OPENMP
            mt19937 r(seed * 1000003u + gen * 7919u + omp_get_thread_num());
            #else
            mt19937 r(seed * 1000003u + gen * 7919u);
            #endif
            #pragma omp for schedule(dynamic, 4)
            for (int i = ELITE; i < POP; i++) {
                const Ind& a = tour(r); const Ind& b = tour(r);
                Perm c = (r() % 1000 < PC * 1000) ? orderCrossover(a.p, b.p, r) : a.p;
                if (r() % 1000 < PM * 1000) mutate(c, r);
                if (r() % 1000 < 100) mutate(c, r);       // occasional double mutation
                nxt[i].p = c;
                nxt[i].s = evaluate(decode(c), S);
            }
        }
        pop.swap(nxt);
        sort(pop.begin(), pop.end(), byFit);

        if (pop[0].s.fit < best.s.fit - 1e-9) { best = pop[0]; stall = 0; }
        else stall++;

        // on stagnation: local search on elite -> reinject (faster convergence)
        if (stall > 0 && stall % 15 == 0) {
            if (localSearch(best, S)) { pop[POP - 1] = best; sort(pop.begin(), pop.end(), byFit); stall = 0; }
        }
        if (gen % 10 == 0 || stall == 0)
            printf("%5d %12.2f %12.2f %12.2f %5d\n", gen, best.s.fit, best.s.first, best.s.count, best.s.len);
    }
    localSearch(best, S);
    printf("[Done] stopped at gen %d. final E[first]=%.2f, E[all]=%.2f, checks=%d\n",
           gen, best.s.first, best.s.count, best.s.len);

    Plan P = decode(best.p);

    // ---- plan structure: unchecked edges = triangle-free graph ----
    bool used[NE] = {};
    for (int k = 0; k < P.len; k++) used[P.e[k]] = true;
    // try 2-coloring to test bipartiteness (K_{a,b})
    int col[NC]; fill(col, col + NC, -1); bool bip = true;
    for (int s = 0; s < NC; s++) if (col[s] < 0) {
        col[s] = 0; queue<int> q; q.push(s);
        while (!q.empty()) { int u = q.front(); q.pop();
            for (int v = 0; v < NC; v++) if (v != u && !used[edgeId[u][v]]) {
                if (col[v] < 0) { col[v] = col[u] ^ 1; q.push(v); } else if (col[v] == col[u]) bip = false; } }
    }
    printf("[Struct] checked edges %d, unchecked edges %d (Mantel bound: checks >= 16)\n", P.len, NE - P.len);
    if (bip) {
        printf("       unchecked-edge graph is bipartite. parts: {");
        for (int v = 0; v < NC; v++) if (col[v] == 0) printf(" %d", v);
        printf(" } | {"); for (int v = 0; v < NC; v++) if (col[v] == 1) printf(" %d", v); printf(" }\n");
    }
    printf("       check order: ");
    for (int k = 0; k < P.len; k++) printf("(%d,%d)%s", EU[P.e[k]], EV[P.e[k]], k + 1 < P.len ? " " : "\n");

    // ---- write plan header (include directly in the user harness) ----
    // for counting: a triangle found at step k is counted only if it was first covered at step k.
    int firstStep[NT];
    for (int k = 0; k < P.len; k++) for (int t = 0; t < NT; t++) if ((P.newCov[k] >> t) & 1) firstStep[t] = k;
    {
        FILE* f = fopen("ga_plan.h", "w");
        fprintf(f, "// check plan found by GA (auto-generated, seed=%u)\n#pragma once\n", seed);
        fprintf(f, "static const int GA_LEN = %d;\n", P.len);
        fprintf(f, "static const int GA_U[GA_LEN] = {"); for (int k = 0; k < P.len; k++) fprintf(f, "%d%s", EU[P.e[k]], k + 1 < P.len ? ", " : "};\n");
        fprintf(f, "static const int GA_V[GA_LEN] = {"); for (int k = 0; k < P.len; k++) fprintf(f, "%d%s", EV[P.e[k]], k + 1 < P.len ? ", " : "};\n");
        fprintf(f, "// step at which triangle {i<j<k} is first decided (dedup for counting all haps)\n");
        fprintf(f, "static const int GA_FIRST[9][9][9] = {\n");
        for (int i = 0; i < NC; i++) { fprintf(f, " {");
            for (int j = 0; j < NC; j++) { fprintf(f, "{");
                for (int k = 0; k < NC; k++) {
                    int v = (i != j && j != k && i != k) ? firstStep[triId[i][j][k]] : -1;
                    fprintf(f, "%d%s", v, k + 1 < NC ? "," : "");
                }
                fprintf(f, "}%s", j + 1 < NC ? "," : ""); }
            fprintf(f, "}%s\n", i + 1 < NC ? "," : ""); }
        fprintf(f, "};\n");
        fclose(f);
    }

    // =================================================================
    // validation: T fresh boards - compare 1:1 with BruteForce + cost model
    // =================================================================
    mt19937 gv(seed ^ 0x9E3779B9u);
    double grSt = 0, bSt = 0, bFailSt = 0, gSt = 0, gFailSt = 0, gCntSt = 0, bCntSt = 0;
    long long bFail = 0, gFail = 0, wrongFirst = 0, wrongCount = 0, wrongBrute = 0;
    bool has[27]; int pos[27];
    for (long long it = 0; it < T; it++) {
        shuffle(deck.begin(), deck.end(), gv);
        uint8_t id[NC];
        memset(has, 0, sizeof(has));
        for (int i = 0; i < NC; i++) { id[i] = deck[i]; has[id[i]] = true; pos[id[i]] = i; }
        int ref = bruteForceCount(id, has);

        auto br = bruteAl(id, has);
        bSt += br.second; bFail += br.first; if (br.first) bFailSt += br.second;
        if ((br.first == 1) != (ref == 0)) wrongBrute++;
        // cost of BruteForce-style full count (all 36 pairs)
        for (int i = 0; i < NC; i++) for (int j = i + 1; j < NC; j++)
            bCntSt += 3 * t_mod + (has[thirdId(id[i], id[j])] ? t_y : t_n);

        // GA plan: find first hap
        double c = 0; int fail = 1;
        for (int k = 0; k < P.len; k++) {
            c += 3 * t_mod;
            if (has[thirdId(id[EU[P.e[k]]], id[EV[P.e[k]]])]) { c += t_y; fail = 0; break; }
            c += t_n;
        }
        gSt += c; gFail += fail; if (fail) gFailSt += c;
        if ((fail == 1) != (ref == 0)) wrongFirst++;

        // GA plan: count all haps (deduplicated)
        int cnt = 0; double cc = 0;
        for (int k = 0; k < P.len; k++) {
            int u = EU[P.e[k]], v = EV[P.e[k]];
            int th = thirdId(id[u], id[v]);
            cc += 3 * t_mod;
            if (has[th]) { cc += t_y; if (firstStep[triId[u][v][pos[th]]] == k) cnt++; }
            else cc += t_n;
        }
        gCntSt += cc;
        if (cnt != ref) wrongCount++;

        // greedy seed plan (for comparison)
        double cg = 0;
        for (int k = 0; k < PG.len; k++) {
            cg += 3 * t_mod;
            if (has[thirdId(id[EU[PG.e[k]]], id[EV[PG.e[k]]])]) { cg += t_y; break; }
            cg += t_n;
        }
        grSt += cg;
    }
    printf("\n===== Validation (%lld fresh boards) =====\n", T);
    printf("%-26s %14s %14s\n", "", "BruteAl", "GA plan");
    printf("%-26s %14.2f %14.2f\n", "avg cost (first hap)", bSt / T, gSt / T);
    printf("%-26s %14lld %14lld\n", "boards with no hap", bFail, gFail);
    printf("%-26s %14.2f %14.2f\n", "avg cost (no-hap boards)", bFail ? bFailSt / bFail : 0, gFail ? gFailSt / gFail : 0);
    printf("%-26s %14.2f %14.2f\n", "avg cost (count all)", bCntSt / T, gCntSt / T);
    printf("%-26s %14lld %14lld\n", "errors (first hap)", wrongBrute, wrongFirst);
    printf("%-26s %14s %14lld\n", "errors (count)", "-", wrongCount);
    printf("first-hap cost reduction: %.2f%%  (greedy seed only: %.2f -> extra GA gain %.2f%%)\n",
           100.0 * (1 - gSt / bSt), grSt / T, 100.0 * (1 - gSt / grSt));

    // =================================================================
    // real benchmark: actual run time (ns/board) on the same boards
    // =================================================================
    {
        const int M = 2000000;
        vector<array<uint8_t, 27>> A(M);   // a,b,c per card (9 cards x 3)
        vector<uint32_t> H(M);             // 27-bit presence mask
        mt19937 gb(seed + 12345);
        for (int m = 0; m < M; m++) {
            shuffle(deck.begin(), deck.end(), gb);
            H[m] = 0;
            for (int i = 0; i < NC; i++) {
                int d = deck[i]; H[m] |= 1u << d;
                A[m][3 * i] = A_(d); A[m][3 * i + 1] = B_(d); A[m][3 * i + 2] = C_(d);
            }
        }
        int PU[NE], PV[NE]; for (int k = 0; k < P.len; k++) { PU[k] = EU[P.e[k]]; PV[k] = EV[P.e[k]]; }
        int L = P.len;
        auto third = [](const uint8_t* a, int i, int j) {
            int x = (6 - a[3*i] - a[3*j]) % 3, y = (6 - a[3*i+1] - a[3*j+1]) % 3, z = (6 - a[3*i+2] - a[3*j+2]) % 3;
            return x * 9 + y * 3 + z;
        };
        volatile long long sink = 0;
        double bestB = 1e18, bestG = 1e18;
        for (int rep = 0; rep < 5; rep++) {
            auto t0 = chrono::steady_clock::now();
            long long s1 = 0;
            for (int m = 0; m < M; m++) {
                const uint8_t* a = A[m].data(); uint32_t h = H[m]; int r = 1;
                for (int i = 0; i < NC && r; i++) for (int j = i + 1; j < NC; j++)
                    if ((h >> third(a, i, j)) & 1) { r = 0; break; }
                s1 += r;
            }
            auto t1 = chrono::steady_clock::now();
            long long s2 = 0;
            for (int m = 0; m < M; m++) {
                const uint8_t* a = A[m].data(); uint32_t h = H[m]; int r = 1;
                for (int k = 0; k < L; k++) if ((h >> third(a, PU[k], PV[k])) & 1) { r = 0; break; }
                s2 += r;
            }
            auto t2 = chrono::steady_clock::now();
            sink += s1 + s2;
            if (s1 != s2) printf("!! benchmark result mismatch\n");
            bestB = min(bestB, chrono::duration<double, nano>(t1 - t0).count() / M);
            bestG = min(bestG, chrono::duration<double, nano>(t2 - t1).count() / M);
        }
        printf("\n===== Timing (%d boards, best of 5) =====\n", M);
        printf("BruteAl : %.2f ns/board\nGA plan : %.2f ns/board  (%.2fx)\n", bestB, bestG, bestB / bestG);
    }
    return 0;
}
