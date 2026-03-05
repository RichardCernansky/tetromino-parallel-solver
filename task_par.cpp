#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <set>
#include <string>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <omp.h>

static const int MAXR = 20;
static const int MAXC = 20;
static const int UNDECIDED = 0;
static const int UNCOVERED = -1;

const int T_VAR[4][4][2] = {
    {{0,0},{0,1},{0,2},{1,1}},
    {{0,0},{1,0},{2,0},{1,1}},
    {{0,1},{1,0},{1,1},{1,2}},
    {{1,0},{0,1},{1,1},{2,1}},
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
    int  undecided_cells;
    int  t_count, z_count;
    int  next_id;
};

struct Best {
    int  board[MAXR][MAXC];
    char piece_type[MAXR*MAXC];
    int  cost;
    int  t_count, z_count;
    int  next_id;
};

struct SharedBest {
    omp_lock_t lock;
    Best solution;
    int cost;
};

static long long g_calls = 0;

std::pair<int,int> first_undecided(const State& s) {
    for (int r = 0; r < s.rows; r++)
        for (int c = 0; c < s.cols; c++)
            if (s.board[r][c] == UNDECIDED)
                return {r, c};
    return {-1, -1};
}

std::vector<Placement> get_placements(const State& s, int r, int c,
                                       const int VAR[4][4][2], char type) {
    std::vector<Placement> result;
    std::set<std::array<std::pair<int,int>,4>> seen;
    for (int v = 0; v < 4; v++) {
        for (int anchor = 0; anchor < 4; anchor++) {
            int ar = VAR[v][anchor][0], ac = VAR[v][anchor][1];
            Placement p; p.type = type; bool valid = true;
            for (int i = 0; i < 4; i++) {
                int row = r + VAR[v][i][0] - ar;
                int col = c + VAR[v][i][1] - ac;
                if (row < 0 || row >= s.rows || col < 0 || col >= s.cols
                    || s.board[row][col] != UNDECIDED) {
                    valid = false; break;
                }
                p.rows[i] = row; p.cols[i] = col;
            }
            if (!valid) continue;
            std::array<std::pair<int,int>,4> key;
            for (int i = 0; i < 4; i++) key[i] = {p.rows[i], p.cols[i]};
            std::sort(key.begin(), key.end());
            if (seen.count(key)) continue;
            seen.insert(key);
            result.push_back(p);
        }
    }
    return result;
}

void apply_piece(State& s, const Placement& p) {
    int id = s.next_id++;
    s.piece_type[id-1] = p.type;
    for (int i = 0; i < 4; i++)
        s.board[p.rows[i]][p.cols[i]] = id;
    s.undecided_cells -= 4;
    if (p.type == 'T') s.t_count++; else s.z_count++;
}

void undo_piece(State& s, const Placement& p) {
    s.next_id--;
    for (int i = 0; i < 4; i++)
        s.board[p.rows[i]][p.cols[i]] = UNDECIDED;
    s.undecided_cells += 4;
    if (p.type == 'T') s.t_count--; else s.z_count--;
}

void apply_uncover(State& s, int r, int c) {
    s.cost += s.weights[r][c];
    s.board[r][c] = UNCOVERED;
    s.undecided_cells--;
}

void undo_uncover(State& s, int r, int c) {
    s.cost -= s.weights[r][c];
    s.board[r][c] = UNDECIDED;
    s.undecided_cells++;
}

// Lock-protected update — the ONLY safe way to update a composite solution
void try_update_best(const State& s, SharedBest& shared) {
    omp_set_lock(&shared.lock);
    if (s.cost < shared.cost) {
        shared.cost = s.cost;
        shared.solution.cost    = s.cost;
        shared.solution.t_count = s.t_count;
        shared.solution.z_count = s.z_count;
        shared.solution.next_id = s.next_id;
        for (int i = 0; i < s.rows; i++)
            for (int j = 0; j < s.cols; j++)
                shared.solution.board[i][j] = s.board[i][j];
        for (int k = 0; k < s.next_id - 1; k++)
            shared.solution.piece_type[k] = s.piece_type[k];
    }
    omp_unset_lock(&shared.lock);
}

// Sequential DFS — called after cutoff, operates in-place with apply/undo
void dfs_seq(State& s, SharedBest& shared) {
    #pragma omp atomic
    g_calls++;

    int best;
    #pragma omp atomic read
    best = shared.cost;
    if (s.cost >= best) return;

    auto [r, c] = first_undecided(s);
    if (r == -1) {
        try_update_best(s, shared);
        return;
    }

    auto t_moves = get_placements(s, r, c, T_VAR, 'T');
    auto z_moves = get_placements(s, r, c, Z_VAR, 'Z');

    for (auto& p : t_moves) {
        apply_piece(s, p);
        dfs_seq(s, shared);
        undo_piece(s, p);
    }
    for (auto& p : z_moves) {
        apply_piece(s, p);
        dfs_seq(s, shared);
        undo_piece(s, p);
    }

    int best2;
    #pragma omp atomic read
    best2 = shared.cost;
    if (s.cost + s.weights[r][c] < best2) {
        apply_uncover(s, r, c);
        dfs_seq(s, shared);
        undo_uncover(s, r, c);
    }
}

// Task-parallel DFS
// Takes State BY VALUE so each task owns its own copy
void dfs_task(State s, SharedBest& shared, int depth, int cutoff) {
    #pragma omp atomic
    g_calls++;

    int best;
    #pragma omp atomic read
    best = shared.cost;
    if (s.cost >= best) return;

    auto [r, c] = first_undecided(s);

    // LEAF — board fully decided
    if (r == -1) {
        try_update_best(s, shared);
        return;   // CRITICAL: without this, falls through to get_placements(-1,-1)
    }

    // Past cutoff — switch to sequential in-place DFS
    if (depth >= cutoff) {
        dfs_seq(s, shared);
        return;
    }

    auto t_moves = get_placements(s, r, c, T_VAR, 'T');
    auto z_moves = get_placements(s, r, c, Z_VAR, 'Z');

    // First child runs locally (saves one State copy),
    // all others are spawned as tasks.
    // Every task pragma has EXPLICIT data-sharing clauses:
    //   firstprivate(child) — task gets its own deep copy of the State
    //   shared(shared)      — all tasks see the same SharedBest

    bool have_local = false;
    State local_child;

    #pragma omp taskgroup
    {
        for (auto& p : t_moves) {
            State child = s;
            apply_piece(child, p);
            if (!have_local) {
                local_child = child;
                have_local = true;
            } else {
                #pragma omp task firstprivate(child) shared(shared)
                dfs_task(child, shared, depth + 1, cutoff);
            }
        }

        for (auto& p : z_moves) {
            State child = s;
            apply_piece(child, p);
            if (!have_local) {
                local_child = child;
                have_local = true;
            } else {
                #pragma omp task firstprivate(child) shared(shared)
                dfs_task(child, shared, depth + 1, cutoff);
            }
        }

        // Uncover branch
        {
            State child = s;
            apply_uncover(child, r, c);
            if (!have_local) {
                local_child = child;
                have_local = true;
            } else {
                #pragma omp task firstprivate(child) shared(shared)
                dfs_task(child, shared, depth + 1, cutoff);
            }
        }

        // Run one child locally in this thread
        if (have_local) {
            dfs_task(local_child, shared, depth + 1, cutoff);
        }
    } // taskgroup barrier: all spawned tasks complete before returning
}

void print_solution(const State& s, const Best& best) {
    std::cout << "\n=== Solution ===\n";
    std::cout << "Cost:     " << best.cost
              << "\nT pieces: " << best.t_count
              << "\nZ pieces: " << best.z_count << "\n\n";
    int w = std::max(3, (int)std::to_string(best.next_id - 1).size() + 1);
    for (int i = 0; i < s.rows; i++) {
        for (int j = 0; j < s.cols; j++) {
            int v = best.board[i][j];
            std::string cell = (v == UNCOVERED)
                ? std::to_string(s.weights[i][j])
                : std::string(1, best.piece_type[v-1]) + std::to_string(v);
            std::cout << std::setw(w) << cell << " ";
        }
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file> [threads] [cutoff]\n";
        return 1;
    }

    int nt  = (argc >= 3) ? std::stoi(argv[2]) : omp_get_max_threads();
    int cut = (argc >= 4) ? std::stoi(argv[3]) : 3;
    omp_set_num_threads(nt);

    std::ifstream fin(argv[1]);
    if (!fin) { std::cerr << "Cannot open: " << argv[1] << "\n"; return 1; }

    State s{};
    fin >> s.rows >> s.cols;
    s.cost = 0; s.undecided_cells = s.rows * s.cols;
    s.t_count = 0; s.z_count = 0; s.next_id = 1;
    for (int i = 0; i < s.rows; i++)
        for (int j = 0; j < s.cols; j++) {
            fin >> s.weights[i][j];
            s.board[i][j] = UNDECIDED;
        }

    int total_weight = 0;
    for (int i = 0; i < s.rows; i++)
        for (int j = 0; j < s.cols; j++)
            total_weight += s.weights[i][j];

    SharedBest shared;
    shared.cost = total_weight;
    omp_init_lock(&shared.lock);
    shared.solution.cost    = total_weight;
    shared.solution.t_count = 0;
    shared.solution.z_count = 0;
    shared.solution.next_id = 1;
    for (int i = 0; i < s.rows; i++)
        for (int j = 0; j < s.cols; j++)
            shared.solution.board[i][j] = UNCOVERED;

    std::cout << "Board:               " << s.rows << " x " << s.cols << "\n";
    std::cout << "Threads:             " << nt << "\n";
    std::cout << "Cutoff depth:        " << cut << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel shared(shared)
    {
        #pragma omp single
        {
            dfs_task(s, shared, 0, cut);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    omp_destroy_lock(&shared.lock);

    std::cout << "Recursive calls:     " << g_calls << "\n";
    std::cout << "Wall time:           " << std::fixed << std::setprecision(3)
              << elapsed << " s\n";
    print_solution(s, shared.solution);
    return 0;
}