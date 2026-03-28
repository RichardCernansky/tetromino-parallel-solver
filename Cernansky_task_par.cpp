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

// States
// kinds: UNDECIDED, UNCOVERED, N > 0   covered by piece id N
static const int MAXR = 20;
static const int MAXC = 20;
static const int UNDECIDED = 0;
static const int UNCOVERED = -1;

const int T_VAR[4][4][2] = {
    {{0,0},{0,1},{0,2},{1,1}},   // T0
    {{0,0},{1,0},{2,0},{1,1}},   // T1
    {{0,1},{1,0},{1,1},{1,2}},   // T2
    {{1,0},{0,1},{1,1},{2,1}},   // T3
};
const int Z_VAR[4][4][2] = {
    {{0,0},{0,1},{1,1},{1,2}},   // Z0
    {{0,1},{1,0},{1,1},{2,0}},   // Z1
    {{0,1},{0,2},{1,0},{1,1}},   // Z2 (S horiz)
    {{0,0},{1,0},{1,1},{2,1}},   // Z3 (S vert)
};

struct Placement {
    int  rows[4], cols[4];
    char type;
};

//  Full board state threaded through the DFS
struct State {
    int  board[MAXR][MAXC];       // cell ownership
    int  weights[MAXR][MAXC];     // fixed input weights
    char piece_type[MAXR*MAXC];   // piece_type[id-1] ->'T' or 'Z'
    int  rows, cols;

    int  cost;           // sum of weights of UNCOVERED cells so far
    int  undecided_sum;  // sum of weights of UNDECIDED cells
    int undecided_cells;
    int  t_count;        // T pieces placed so far
    int  z_count;        // Z pieces placed so far
    int  next_id;        // next piece id (1-based)
};

//  stores best solution sofar
struct Best {
    int  board[MAXR][MAXC];
    char piece_type[MAXR*MAXC];
    int  cost;
    int  t_count, z_count;
    int  next_id;        // how many pieces were placed
};

struct SharedBest {
    alignas(64) int cost;
    char pad1[64];
    alignas(64) omp_lock_t lock;
    Best solution;
    bool found_optimal;
};

static long long g_calls = 0;

//  Find first UNDECIDED cell in row-major order.
//  Returns {-1,-1} when the board is fully decided.
std::pair<int,int> first_undecided(const State& s)
{
    for (int r = 0; r < s.rows; r++)
        for (int c = 0; c < s.cols; c++)
            if (s.board[r][c] == UNDECIDED)
                return {r, c};
    return {-1, -1};
}

//  Generate all valid placements of one piece
//  type that cover cell (r, c).
//  For every variant, treat each of its 4 cells as the anchor landing on (r,c).
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
    // handle ids
    int id = s.next_id++; s.piece_type[id-1] = p.type;
    // go through placement cells and update board
    for (int i = 0; i < 4; i++) {
        s.undecided_sum -= s.weights[p.rows[i]][p.cols[i]];
        s.board[p.rows[i]][p.cols[i]] = id;
    }
    // update undecided
    s.undecided_cells -= 4;
    if (p.type == 'T') s.t_count++; else s.z_count++;
}

// reverse of apply piece
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
    // add to weights and subtract from undecided_sum
    s.cost += s.weights[r][c]; s.undecided_sum -= s.weights[r][c];
    // apply uncovered in board
    s.board[r][c] = UNCOVERED; s.undecided_cells--;
}

//reverse of apply uncover
void undo_uncover(State& s, int r, int c) {
    s.cost -= s.weights[r][c]; s.undecided_sum += s.weights[r][c];
    s.board[r][c] = UNDECIDED; s.undecided_cells++;
}

// Parity prune
//  Returns true  -> this branch CANNOT satisfy
//                  the parity constraint → prune.
//  Returns false -> parity is still satisfiable.
bool parity_prune(int t_count, int z_count, int undecided_cells)
{
    int diff = t_count - z_count;   // positive: more T placed
    if (diff < 0) diff = -diff;     // |diff|

    // Maximum additional pieces we could place
    int max_more = undecided_cells / 4;

    return (diff > max_more + 1);
}


//  Compute trivial lower bound:
//    k = (rows * cols) mod 4
//    lb = sum of k smallest weights on the board
//  If k == 0, lb = 0.
int trivial_lower_bound(const State& s)
{
    int k = (s.rows * s.cols) % 4;
    if (k == 0) return 0;

    std::vector<int> w;
    w.reserve(s.rows * s.cols);
    for (int i = 0; i < s.rows; i++)
        for (int j = 0; j < s.cols; j++)
            w.push_back(s.weights[i][j]);
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

// sequential version of dfs_task with atomic reads on shared and locking for shared writes
void dfs_seq(State& s, SharedBest& shared, int lb) {
    #pragma omp atomic
    g_calls++;

    {
        bool opt;
        #pragma omp atomic read
        opt = shared.found_optimal;
        if (opt) return;
    }

    {
        int best;
        #pragma omp atomic read
        best = shared.cost;
        if (s.cost >= best) return;
    }

    if (parity_prune(s.t_count, s.z_count, s.undecided_cells)) return;

    auto [r,c] = first_undecided(s);
    if (r == -1) {
        int owned = s.cost;

        int old;
        #pragma omp atomic read
        old = shared.cost;

        if (owned < old) {                      // cheap check before locking
            omp_set_lock(&shared.lock);
            int old_2 = shared.cost;            // re-check inside lock
            if (owned < old_2) {
                shared.cost = owned;
                shared.solution.cost      = s.cost;
                shared.solution.t_count   = s.t_count;
                shared.solution.z_count   = s.z_count;
                shared.solution.next_id   = s.next_id;
                for (int i = 0; i < s.rows; i++)
                    for (int j = 0; j < s.cols; j++)
                        shared.solution.board[i][j] = s.board[i][j];
                for (int k = 0; k < s.next_id - 1; k++)
                    shared.solution.piece_type[k] = s.piece_type[k];
                if (owned == lb)
                    shared.found_optimal = true;
            }
            omp_unset_lock(&shared.lock);
        }
        return;
    }


    auto t_moves = get_placements(s,r,c,T_VAR,'T');
    auto z_moves = get_placements(s,r,c,Z_VAR,'Z');
    auto cov = [&](const Placement& p){ int s2=0; for(int i=0;i<4;i++) s2+=s.weights[p.rows[i]][p.cols[i]]; return s2; };

    std::vector<Placement> all;
    for (auto& p : t_moves) all.push_back(p);
    for (auto& p : z_moves) all.push_back(p);
    std::sort(all.begin(),all.end(),[&](const Placement& a,const Placement& b){return cov(a)>cov(b);});

    for (auto& p : all) {
        bool opt;
        #pragma omp atomic read
        opt = shared.found_optimal;
        if (opt) return;
        apply_piece(s,p);
        dfs_seq(s,shared,lb);
        undo_piece(s,p);
    }
    bool opt;
    #pragma omp atomic read
    opt = shared.found_optimal;
    if (opt) return;

    // now spawn new task
    int old_2;
    // get always correct atomic value, most recent one from shared-global  memory
    # pragma omp atomic read
    old_2 = shared.cost;
    // Uncover branch - add task
    if (s.cost + s.weights[r][c] < old_2) {
        apply_uncover(s,r,c);
        dfs_seq(s,shared,lb);
        undo_uncover(s,r,c);
    }
}


void dfs_task(State s, SharedBest& shared, int lb, int depth, int cutoff) {

    #pragma omp atomic
    g_calls++;

    bool opt;
    #pragma omp atomic read
    opt = shared.found_optimal;
    if (opt) return;

    int best;
    #pragma omp atomic read
    best = shared.cost;
    if (s.cost >= best) return;

    if (parity_prune(s.t_count, s.z_count, s.undecided_cells)) return;

    auto rc = first_undecided(s);
    int r = rc.first;
    int c = rc.second;
    if (r == -1) { // if board fully decided
        int owned = s.cost;

        int old;
        # pragma omp atomic read
        old = shared.cost;
        // first check if its even worth locking: if own < old -> update
        if (owned < old) {
            // microseconds pass
            // acquire lock
            omp_set_lock(&shared.lock);
            int old_2;
            old_2 = shared.cost;
            if (owned < old_2) {
                //shared.cost stored with release atomic
                shared.cost = owned;
                // update shared.solution with copies - already locked
                shared.solution.cost = s.cost;
                shared.solution.t_count = s.t_count;
                shared.solution.z_count = s.z_count;
                shared.solution.next_id = s.next_id;
                for (int i = 0; i < s.rows; i++) for (int j = 0; j < s.cols; j++) shared.solution.board[i][j] = s.board[i][j];
                for (int k = 0; k < s.next_id - 1; k++) shared.solution.piece_type[k] = s.piece_type[k];
                if (owned == lb) shared.found_optimal = true;
            }
            // release lock
            omp_unset_lock(&shared.lock);
        }
        return;
    }

    // start doing seq after this level
    if (depth >= cutoff) {
        dfs_seq(s, shared, lb);
        return;
    }

    auto t_moves = get_placements(s, r, c, T_VAR, 'T');
    auto z_moves = get_placements(s, r, c, Z_VAR, 'Z');

    auto cov = [&](const Placement& p) {
        int sum = 0;
        for (int i = 0; i < 4; i++) sum += s.weights[p.rows[i]][p.cols[i]];
        return sum;
    };

    // get all placements and sort them decreasingly by the coverage they provide -> optimal faster
    std::vector<Placement> all;
    for (auto& p : t_moves) all.push_back(p);
    for (auto& p : z_moves) all.push_back(p);
    std::sort(all.begin(), all.end(), [&](const Placement& a, const Placement& b) {
        return cov(a) > cov(b);
    });


    bool has_local = false;
    State local_child;

    int cur_best;
    #pragma omp atomic read
    cur_best = shared.cost;
    bool can_uncover = (s.cost + s.weights[r][c] < cur_best);

    for (int idx = 0; idx < (int)all.size(); idx++) {
    // recursively spawn new task in each iteration for all the possible placements
        bool stop;
        #pragma omp atomic read
        stop = shared.found_optimal;
        if (stop) break;

        // create new copy unique for the thread
        State child = s;
        apply_piece(child, all[idx]);

        if (!has_local) {
            local_child = child;
            has_local = true;
        } else {

            // idle thread from the team picks this up
            #pragma omp task firstprivate(child) shared(shared)
            {
                dfs_task(child, shared, lb, depth + 1, cutoff);
            }
        }
    }


    // recursively spawn task for the uncovered shared
    if (can_uncover) {
        bool stop;
        #pragma omp atomic read
        stop = shared.found_optimal;

        if (!stop) {
            // need copy before applying uncover so it's not destroyed
            State child = s;
            apply_uncover(child, r, c);

            if (!has_local) {
                local_child = child;
                has_local = true;
            } else {
                // idle thread from the team picks this up, firstprivate - private copy of the state so they don't race, explicitly sharing shared
                #pragma omp task firstprivate(child) shared(shared)
                {
                    dfs_task(child, shared, lb, depth + 1, cutoff);
                }
            }
        }
    }

    // if there is local
    if (has_local) {
        dfs_task(local_child, shared, lb, depth + 1, cutoff);
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file> [threads] [cutoff]\n";
        return 1;
    }

    // int nt = (argc >= 3) ? std::stoi(argv[2]) : omp_get_max_threads();
    int nt = std::stoi(argv[2]);
    int cut =  std::stoi(argv[3]);
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
    shared.cost = s.undecided_sum;
    shared.found_optimal = false;
    omp_init_lock(&shared.lock);

    shared.solution.cost = s.undecided_sum;
    shared.solution.t_count = 0;
    shared.solution.z_count = 0;
    shared.solution.next_id = 1;

    for (int i = 0; i < s.rows; i++) {
        for (int j = 0; j < s.cols; j++) {
            shared.solution.board[i][j] = UNCOVERED;
        }
    }

    std::cout << "Board:              " << s.rows << " x " << s.cols << "\n";
    std::cout << "Trivial lower bound: " << lb << "\n";
    std::cout << "Threads:            " << nt << "\n";
    std::cout << "Cutoff depth:       " << cut << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();


    // parallel processing - create team of threads, explicitly sharing variable shared SharedBest struct
    #pragma omp parallel shared(shared)
        {
        #pragma omp single
            {
                dfs_task(s, shared, lb, 0, cut);
            }
            // implicit barrier here – all tasks spawned in the dfs_task must complete before we exit
        }


    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    omp_destroy_lock(&shared.lock);

    std::cout << "Recursive calls:    " << g_calls << "\n";
    std::cout << "Wall time:          " << std::fixed << std::setprecision(3) << elapsed << " s\n";
    std::cout << (shared.found_optimal ? "Result: OPTIMAL\n" : "Result: best found\n");
    print_solution(s, shared.solution);
    return 0;
}