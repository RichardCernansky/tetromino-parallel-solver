#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <set>
#include <string>
#include <climits>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <omp.h>

static const int MAXR = 50;
static const int MAXC = 50;
static const int UNDECIDED = 0;
static const int UNCOVERED = -1;

const int T_VAR[4][4][2] = {
    {{0,0},{0,1},{0,2},{1,1}},
    {{0,0},{1,0},{2,0},{1,1}},
    {{0,1},{1,0},{1,1},{1,2}},
    {{0,0},{0,1},{1,1},{2,1}},
};

const int Z_VAR[4][4][2] = {
    {{0,0},{0,1},{1,1},{1,2}},
    {{0,1},{1,0},{1,1},{2,0}},
    {{0,1},{0,2},{1,0},{1,1}},
    {{0,0},{1,0},{1,1},{2,1}},
};

struct Placement {
    int  rows[4], cols[4];
    char type;
};

struct State {
    int  board[MAXR][MAXC];
    int  weights[MAXR][MAXC];
    char piece_type[MAXR*MAXC];
    int  rows, cols;
    int  cost;
    int  undecided_sum;
    int  undecided_cells;
    int  t_count, z_count, next_id;
};

struct Best {
    int  board[MAXR][MAXC];
    char piece_type[MAXR*MAXC];
    int  cost, t_count, z_count, next_id;
};

struct SharedBest {
    alignas(64) std::atomic<int> cost;
    char pad1[64];
    alignas(64) omp_lock_t lock;
    Best solution;
    std::atomic<bool> found_optimal;
};

static std::atomic<long long> g_calls{0};

std::pair<int,int> first_undecided(const State& s) {
    for (int r = 0; r < s.rows; r++)
        for (int c = 0; c < s.cols; c++)
            if (s.board[r][c] == UNDECIDED) return {r, c};
    return {-1, -1};
}

std::vector<Placement> get_placements(const State& s, int r, int c, const int VAR[4][4][2], char type) {
    std::vector<Placement> result;
    std::set<std::array<std::pair<int,int>,4>> seen;
    for (int v = 0; v < 4; v++) {
        for (int anchor = 0; anchor < 4; anchor++) {
            int ar = VAR[v][anchor][0], ac = VAR[v][anchor][1];
            Placement p; p.type = type; bool valid = true;
            for (int i = 0; i < 4; i++) {
                int row = r + VAR[v][i][0] - ar, col = c + VAR[v][i][1] - ac;
                if (row < 0 || row >= s.rows || col < 0 || col >= s.cols || s.board[row][col] != UNDECIDED) {
                    valid = false; break;
                }
                p.rows[i] = row; p.cols[i] = col;
            }
            if (!valid) continue;
            std::array<std::pair<int,int>,4> key;
            for (int i = 0; i < 4; i++) key[i] = {p.rows[i], p.cols[i]};
            std::sort(key.begin(), key.end());
            if (seen.count(key)) continue;
            seen.insert(key); result.push_back(p);
        }
    }
    return result;
}

void apply_piece(State& s, const Placement& p) {
    int id = s.next_id++; s.piece_type[id-1] = p.type;
    for (int i = 0; i < 4; i++) {
        s.undecided_sum -= s.weights[p.rows[i]][p.cols[i]];
        s.board[p.rows[i]][p.cols[i]] = id;
    }
    s.undecided_cells -= 4;
    if (p.type == 'T') s.t_count++; else s.z_count++;
}

void undo_piece(State& s, const Placement& p) {
    s.next_id--;
    for (int i = 0; i < 4; i++) {
        s.undecided_sum += s.weights[p.rows[i]][p.cols[i]];
        s.board[p.rows[i]][p.cols[i]] = UNDECIDED;
    }
    s.undecided_cells += 4;
    if (p.type == 'T') s.t_count--; else s.z_count--;
}

void apply_uncover(State& s, int r, int c) {
    s.cost += s.weights[r][c]; s.undecided_sum -= s.weights[r][c];
    s.board[r][c] = UNCOVERED; s.undecided_cells--;
}

void undo_uncover(State& s, int r, int c) {
    s.cost -= s.weights[r][c]; s.undecided_sum += s.weights[r][c];
    s.board[r][c] = UNDECIDED; s.undecided_cells++;
}

bool parity_prune(int t, int z, int u) {
    int d = (t - z < 0) ? z - t : t - z;
    return d > u / 4 + 1;
}

int trivial_lower_bound(const State& s) {
    int k = (s.rows * s.cols) % 4;
    if (k == 0) return 0;
    std::vector<int> w;
    for (int i = 0; i < s.rows; i++)
        for (int j = 0; j < s.cols; j++) w.push_back(s.weights[i][j]);
    std::sort(w.begin(), w.end());
    int lb = 0;
    for (int i = 0; i < k; i++) lb += w[i];
    return lb;
}

void print_solution(const State& s, const Best& best) {
    std::cout << "\n=== Solution ===\n";
    std::cout << "Cost:     " << best.cost << "\nT pieces: " << best.t_count << "\nZ pieces: " << best.z_count << "\n\n";
    int w = std::max(3, (int)std::to_string(best.next_id - 1).size() + 1);
    for (int i = 0; i < s.rows; i++) {
        for (int j = 0; j < s.cols; j++) {
            int v = best.board[i][j];
            std::string cell = (v == UNCOVERED) ? std::to_string(s.weights[i][j]) : std::string(1, best.piece_type[v - 1]) + std::to_string(v);
            std::cout << std::setw(w) << cell << " ";
        }
        std::cout << "\n";
    }
}

void generate_tasks(State s, int depth, int cutoff, int lb, std::vector<State>& pool) {
    if (depth >= cutoff) { pool.push_back(s); return; }
    auto [r, c] = first_undecided(s);
    if (r == -1) return;
    auto t_moves = get_placements(s, r, c, T_VAR, 'T');
    auto z_moves = get_placements(s, r, c, Z_VAR, 'Z');
    for (auto& p : t_moves) { apply_piece(s, p); generate_tasks(s, depth+1, cutoff, lb, pool); undo_piece(s, p); }
    for (auto& p : z_moves) { apply_piece(s, p); generate_tasks(s, depth+1, cutoff, lb, pool); undo_piece(s, p); }
}

void dfs_parallel(State& s, SharedBest& shared, int lb) {
    g_calls.fetch_add(1, std::memory_order_relaxed);
    if (shared.found_optimal.load(std::memory_order_acquire)) return;
    int best = shared.cost.load(std::memory_order_acquire);
    if (s.cost >= best) return;
    if (parity_prune(s.t_count, s.z_count, s.undecided_cells)) return;

    auto [r, c] = first_undecided(s);
    if (r == -1) {
        int my = s.cost, old = shared.cost.load(std::memory_order_acquire);
        if (my < old) {
            omp_set_lock(&shared.lock);
            if (my < shared.cost.load(std::memory_order_relaxed)) {
                shared.cost.store(my, std::memory_order_release);
                shared.solution.cost = s.cost; shared.solution.t_count = s.t_count; shared.solution.z_count = s.z_count; shared.solution.next_id = s.next_id;
                for (int i = 0; i < s.rows; i++) for (int j = 0; j < s.cols; j++) shared.solution.board[i][j] = s.board[i][j];
                for (int k = 0; k < s.next_id - 1; k++) shared.solution.piece_type[k] = s.piece_type[k];
                if (my == lb) shared.found_optimal.store(true, std::memory_order_release);
            }
            omp_unset_lock(&shared.lock);
        }
        return;
    }

    auto t_moves = get_placements(s, r, c, T_VAR, 'T');
    auto z_moves = get_placements(s, r, c, Z_VAR, 'Z');
    auto cov = [&](const Placement& p) { int sum = 0; for (int i = 0; i < 4; i++) sum += s.weights[p.rows[i]][p.cols[i]]; return sum; };
    std::vector<Placement> all;
    for (auto& p : t_moves) all.push_back(p);
    for (auto& p : z_moves) all.push_back(p);
    std::sort(all.begin(), all.end(), [&](const Placement& a, const Placement& b) { return cov(a) > cov(b); });

    for (auto& p : all) {
        if (shared.found_optimal.load(std::memory_order_acquire)) return;
        apply_piece(s, p); dfs_parallel(s, shared, lb); undo_piece(s, p);
    }

    if (shared.found_optimal.load(std::memory_order_acquire)) return;
    int best2 = shared.cost.load(std::memory_order_acquire);
    if (s.cost + s.weights[r][c] < best2) {
        apply_uncover(s, r, c); dfs_parallel(s, shared, lb); undo_uncover(s, r, c);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file> [threads] [cutoff]\n";
        return 1;
    }

    int nt = (argc >= 3) ? std::stoi(argv[2]) : omp_get_max_threads();
    int cut = (argc >= 4) ? std::stoi(argv[3]) : 3;
    omp_set_num_threads(nt);

    std::ifstream fin(argv[1]);
    if (!fin) { std::cerr << "Cannot open: " << argv[1] << "\n"; return 1; }

    State s{};
    fin >> s.rows >> s.cols;
    s.cost = 0; s.undecided_sum = 0; s.undecided_cells = s.rows * s.cols;
    s.t_count = 0; s.z_count = 0; s.next_id = 1;
    for (int i = 0; i < s.rows; i++)
        for (int j = 0; j < s.cols; j++) {
            fin >> s.weights[i][j]; s.board[i][j] = UNDECIDED;
            s.undecided_sum += s.weights[i][j];
        }

    int lb = trivial_lower_bound(s);
    SharedBest shared;
    shared.cost.store(s.undecided_sum, std::memory_order_relaxed);
    shared.found_optimal.store(false, std::memory_order_relaxed);
    omp_init_lock(&shared.lock);
    shared.solution.cost = s.undecided_sum;

    std::cout << "Board:              " << s.rows << " x " << s.cols << "\n";
    std::cout << "Trivial lower bound: " << lb << "\n";
    std::cout << "Threads:            " << nt << "\n";
    std::cout << "Cutoff depth:       " << cut << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<State> pool;
    generate_tasks(s, 0, cut, lb, pool);
    std::cout << "Generated tasks:    " << pool.size() << "\n";
    std::cout << "Starting parallel search...\n";

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (size_t i = 0; i < pool.size(); i++) {
                #pragma omp task firstprivate(i)
                {
                    State ts = pool[i]; // each task makes copy of the state from the pool - avoid conflict
                    dfs_parallel(ts, shared, lb);
                }
            }
            #pragma omp taskwait
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    omp_destroy_lock(&shared.lock);

    std::cout << "Recursive calls:    " << g_calls.load() << "\n";
    std::cout << "Wall time:          " << std::fixed << std::setprecision(3) << elapsed << " s\n";
    std::cout << (shared.found_optimal.load() ? "Result: OPTIMAL\n" : "Result: best found\n");
    print_solution(s, shared.solution);
    return 0;
}