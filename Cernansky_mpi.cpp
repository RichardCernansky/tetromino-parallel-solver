// ============================================================================
// SQM Quatromino Tiling Solver — MPI + OpenMP Hybrid (Master-Slave)
// ============================================================================
//
// Architecture overview:
//   Rank 0 (Master):
//     1. Reads input and generates a pool of partial DFS states (sequential)
//     2. Dynamically distributes batches of states to worker ranks
//     3. Receives results, updates global best, sends more work or termination
//
//   Ranks 1..P-1 (Workers):
//     1. Read the same input file (avoids broadcasting large weight matrix)
//     2. Receive batches of states from master
//     3. Process them using OpenMP parallel-for with schedule(dynamic,1)
//     4. Report local best solution back to master
//
//   Within each worker, OpenMP threads share a process-local SharedBest
//   protected by an OMP lock — identical to the pure-OpenMP version.
//
//   The master piggybacks the current global best cost with every work
//   assignment so workers can initialise tighter pruning bounds.
//
// Compilation (on cluster-arm.in.fit.cvut.cz):
//   mpicxx -O2 -fopenmp -std=c++17 -o sqm_mpi sqm_mpi.cpp
//
// Execution:
//   mpirun -np <processes> ./sqm_mpi <mapfile> [threads_per_proc] [cutoff]
//
//   Example (4 MPI processes, 4 OpenMP threads each, cutoff depth 4):
//   mpirun -np 4 ./sqm_mpi mapa5_11.txt 4 4
// ============================================================================

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
#include <cstring>

#include <mpi.h>
#include <omp.h>

// ============================================================================
// Constants and piece definitions
// ============================================================================

static const int MAXR = 20;          // maximum board rows
static const int MAXC = 20;          // maximum board columns
static const int UNDECIDED = 0;      // cell not yet processed
static const int UNCOVERED = -1;     // cell explicitly left uncovered

// --- T-piece: 4 rotations, each with 4 cells (row,col offsets) ---
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

// ============================================================================
// MPI protocol constants
// ============================================================================

static const int TAG_WORK  = 10;     // master -> worker: here is work
static const int TAG_RESULT = 20;    // worker -> master: here is my result

// ============================================================================
// Data structures
// ============================================================================

// Describes one placed piece by its 4 cell coordinates and type
struct Placement {
    int  rows[4], cols[4];
    char type;                       // 'T' or 'Z'
};

// Full DFS state threaded through the search tree
struct State {
    int  board[MAXR][MAXC];          // cell ownership (0=undecided, -1=uncovered, >0=piece id)
    int  weights[MAXR][MAXC];        // input weights (constant after initialisation)
    char piece_type[MAXR*MAXC];      // piece_type[id-1] -> 'T' or 'Z'
    int  rows, cols;                 // board dimensions

    int  cost;                       // sum of weights of UNCOVERED cells so far
    int  undecided_sum;              // sum of weights of remaining UNDECIDED cells
    int  undecided_cells;            // count of remaining UNDECIDED cells
    int  t_count;                    // T pieces placed
    int  z_count;                    // Z pieces placed
    int  next_id;                    // next piece id (1-based)
};

// Stores the best complete solution found so far
struct Best {
    int  board[MAXR][MAXC];
    char piece_type[MAXR*MAXC];
    int  cost;
    int  t_count, z_count;
    int  next_id;
};

// Process-local shared best, protected by OMP lock for thread safety
struct SharedBest {
    alignas(64) int cost;            // best cost found (read atomically by DFS threads)
    char pad1[64];                   // padding to avoid false sharing
    alignas(64) omp_lock_t lock;     // lock for updating the full solution
    Best solution;
    bool found_optimal;              // true if cost == trivial lower bound
};

// --------------------------------------------------------------------------
// Serialised state for MPI transfer.
// Excludes 'weights' since all ranks read the input file independently.
// --------------------------------------------------------------------------
struct MPIState {
    int  board[MAXR][MAXC];
    char piece_type[MAXR*MAXC];
    int  cost;
    int  undecided_sum;
    int  undecided_cells;
    int  t_count;
    int  z_count;
    int  next_id;
};

// Result sent from worker back to master
struct WorkerResult {
    int  best_cost;                  // best cost found by this worker batch
    int  board[MAXR][MAXC];          // board of best solution
    char piece_type[MAXR*MAXC];
    int  t_count, z_count, next_id;
    bool found_optimal;
};

// ============================================================================
// Packing / unpacking helpers for MPI transport
// ============================================================================

// Pack a State into an MPIState (drops constant weights/dims)
MPIState pack_state(const State& s) {
    MPIState ms;
    std::memcpy(ms.board, s.board, sizeof(ms.board));
    std::memcpy(ms.piece_type, s.piece_type, sizeof(ms.piece_type));
    ms.cost            = s.cost;
    ms.undecided_sum   = s.undecided_sum;
    ms.undecided_cells = s.undecided_cells;
    ms.t_count         = s.t_count;
    ms.z_count         = s.z_count;
    ms.next_id         = s.next_id;
    return ms;
}

// Unpack an MPIState into a full State, copying weights/dims from `base`
State unpack_state(const MPIState& ms, const State& base) {
    State s;
    // Copy constant data from base (weights, dimensions)
    std::memcpy(s.weights, base.weights, sizeof(s.weights));
    s.rows = base.rows;
    s.cols = base.cols;
    // Copy variable data from MPIState
    std::memcpy(s.board, ms.board, sizeof(s.board));
    std::memcpy(s.piece_type, ms.piece_type, sizeof(s.piece_type));
    s.cost            = ms.cost;
    s.undecided_sum   = ms.undecided_sum;
    s.undecided_cells = ms.undecided_cells;
    s.t_count         = ms.t_count;
    s.z_count         = ms.z_count;
    s.next_id         = ms.next_id;
    return s;
}

// ============================================================================
// Core solver functions (unchanged from the OpenMP version)
// ============================================================================

// Per-rank recursive call counter (reduced across MPI at the end)
static long long g_calls = 0;

// Find the first UNDECIDED cell in row-major order.
// Returns {-1,-1} when the board is fully decided.
std::pair<int,int> first_undecided(const State& s)
{
    for (int r = 0; r < s.rows; r++)
        for (int c = 0; c < s.cols; c++)
            if (s.board[r][c] == UNDECIDED)
                return {r, c};
    return {-1, -1};
}

// Generate all valid placements of one piece type that cover cell (r,c).
// For every rotation variant, each of the 4 cells is tried as anchor on (r,c).
// Deduplication via sorted coordinate key prevents exploring mirror placements.
std::vector<Placement> get_placements(const State& s, int r, int c,
                                       const int VAR[4][4][2], char type)
{
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

// Place a piece on the board, updating all bookkeeping
void apply_piece(State& s, const Placement& p) {
    int id = s.next_id++;
    s.piece_type[id-1] = p.type;
    for (int i = 0; i < 4; i++) {
        s.undecided_sum -= s.weights[p.rows[i]][p.cols[i]];
        s.board[p.rows[i]][p.cols[i]] = id;
    }
    s.undecided_cells -= 4;
    if (p.type == 'T') s.t_count++; else s.z_count++;
}

// Reverse of apply_piece (backtracking)
void undo_piece(State& s, const Placement& p) {
    s.next_id--;
    for (int i = 0; i < 4; i++) {
        s.undecided_sum += s.weights[p.rows[i]][p.cols[i]];
        s.board[p.rows[i]][p.cols[i]] = UNDECIDED;
    }
    s.undecided_cells += 4;
    if (p.type == 'T') s.t_count--; else s.z_count--;
}

// Mark a cell as uncovered (adds its weight to cost)
void apply_uncover(State& s, int r, int c) {
    s.cost += s.weights[r][c];
    s.undecided_sum -= s.weights[r][c];
    s.board[r][c] = UNCOVERED;
    s.undecided_cells--;
}

// Reverse of apply_uncover (backtracking)
void undo_uncover(State& s, int r, int c) {
    s.cost -= s.weights[r][c];
    s.undecided_sum += s.weights[r][c];
    s.board[r][c] = UNDECIDED;
    s.undecided_cells++;
}

// Returns true if the T/Z balance constraint can no longer be satisfied.
// |t_count - z_count| must be <= (remaining pieces placeable + 1).
bool parity_prune(int t_count, int z_count, int undecided_cells)
{
    int diff = std::abs(t_count - z_count);
    int max_more = undecided_cells / 4;
    return (diff > max_more + 1);
}

// Trivial lower bound: sum of k smallest weights, where k = (rows*cols) mod 4.
// If k == 0, a perfect tiling may exist and lb = 0.
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

// ============================================================================
// State pool generation (sequential BFS to cutoff depth)
// ============================================================================
//
// Expands the search tree from the root to `cutoff` levels deep.
// Each leaf of this expansion becomes a work unit in the pool.
// No pruning is applied here to keep the pool balanced; the DFS
// will prune when it processes each state.

void generate_states(State s, int depth, int cutoff, int lb,
                     std::vector<State>& pool)
{
    if (depth >= cutoff) {
        pool.push_back(s);
        return;
    }
    auto [r, c] = first_undecided(s);
    if (r == -1) return;            // board fully decided — nothing to add

    // Try all T-piece placements covering (r,c)
    auto t_moves = get_placements(s, r, c, T_VAR, 'T');
    for (auto& p : t_moves) {
        apply_piece(s, p);
        generate_states(s, depth+1, cutoff, lb, pool);
        undo_piece(s, p);
    }

    // Try all Z-piece placements covering (r,c)
    auto z_moves = get_placements(s, r, c, Z_VAR, 'Z');
    for (auto& p : z_moves) {
        apply_piece(s, p);
        generate_states(s, depth+1, cutoff, lb, pool);
        undo_piece(s, p);
    }

    // Try leaving (r,c) uncovered
    apply_uncover(s, r, c);
    generate_states(s, depth+1, cutoff, lb, pool);
    undo_uncover(s, r, c);
}

// ============================================================================
// DFS with branch-and-bound (runs within OpenMP threads on each worker)
// ============================================================================
//
// Identical to the pure-OpenMP version: uses SharedBest (process-local)
// with OMP atomics for the cost check and an OMP lock for solution update.

void dfs(State& s, SharedBest& shared, int lb)
{
    // Count this recursive call (per-rank counter, reduced later)
    #pragma omp atomic
    g_calls++;

    // --- Early termination if optimal already found ---
    {
        bool opt;
        #pragma omp atomic read
        opt = shared.found_optimal;
        if (opt) return;
    }

    // --- Cost-bound pruning: current cost already >= best known ---
    int best;
    #pragma omp atomic read
    best = shared.cost;
    if (s.cost >= best) return;

    // --- Parity pruning: T/Z balance no longer achievable ---
    if (parity_prune(s.t_count, s.z_count, s.undecided_cells)) return;

    // --- Find next cell to decide ---
    auto [r, c] = first_undecided(s);

    if (r == -1) {
        // Board fully decided — this is a complete tiling
        int owned = s.cost;
        int old;
        #pragma omp atomic read
        old = shared.cost;

        if (owned < old) {
            // Lock and double-check before updating (another thread may
            // have improved the bound between the atomic read and now)
            omp_set_lock(&shared.lock);
            if (owned < shared.cost) {
                shared.cost = owned;
                shared.solution.cost = s.cost;
                shared.solution.t_count = s.t_count;
                shared.solution.z_count = s.z_count;
                shared.solution.next_id = s.next_id;
                for (int i = 0; i < s.rows; i++)
                    for (int j = 0; j < s.cols; j++)
                        shared.solution.board[i][j] = s.board[i][j];
                for (int k = 0; k < s.next_id - 1; k++)
                    shared.solution.piece_type[k] = s.piece_type[k];
                if (owned == lb) shared.found_optimal = true;
            }
            omp_unset_lock(&shared.lock);
        }
        return;
    }

    // --- Generate and sort placements (higher-weight-covered first) ---
    auto t_moves = get_placements(s, r, c, T_VAR, 'T');
    auto z_moves = get_placements(s, r, c, Z_VAR, 'Z');

    auto cov = [&](const Placement& p) {
        int sum = 0;
        for (int i = 0; i < 4; i++) sum += s.weights[p.rows[i]][p.cols[i]];
        return sum;
    };

    std::vector<Placement> all;
    all.reserve(t_moves.size() + z_moves.size());
    for (auto& p : t_moves) all.push_back(p);
    for (auto& p : z_moves) all.push_back(p);
    // Descending sort: placements that cover the most weight first
    // → finds good solutions quickly → tightens pruning bound early
    std::sort(all.begin(), all.end(),
              [&](const Placement& a, const Placement& b) {
                  return cov(a) > cov(b);
              });

    // --- Try each placement ---
    for (auto& p : all) {
        {
            bool opt;
            #pragma omp atomic read
            opt = shared.found_optimal;
            if (opt) return;
        }
        apply_piece(s, p);
        dfs(s, shared, lb);
        undo_piece(s, p);
    }

    // --- Try leaving (r,c) uncovered ---
    {
        bool opt;
        #pragma omp atomic read
        opt = shared.found_optimal;
        if (opt) return;
    }
    int old_best;
    #pragma omp atomic read
    old_best = shared.cost;
    // Only recurse if uncovering this cell still allows improvement
    if (s.cost + s.weights[r][c] < old_best) {
        apply_uncover(s, r, c);
        dfs(s, shared, lb);
        undo_uncover(s, r, c);
    }
}

// ============================================================================
// Pretty-print the solution
// ============================================================================

void print_solution(const State& s, const Best& best)
{
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
                : std::string(1, best.piece_type[v - 1]) + std::to_string(v);
            std::cout << std::setw(w) << cell << " ";
        }
        std::cout << "\n";
    }
}

// ============================================================================
// Main — MPI Master-Slave orchestration
// ============================================================================
//
// Protocol:
//   Master -> Worker:  header[2] = {batch_count, global_best_cost}
//                      If batch_count == 0  →  terminate.
//                      Else: followed by batch_count × MPIState structs.
//
//   Worker -> Master:  WorkerResult struct (always sent after each batch).

int main(int argc, char* argv[])
{
    // ---- MPI initialisation with thread support ----
    // MPI_THREAD_FUNNELED: only the main thread makes MPI calls.
    // OpenMP threads run DFS but never call MPI directly.
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "MPI does not support MPI_THREAD_FUNNELED\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // ---- Parse command-line arguments ----
    if (argc < 2) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0]
                      << " <file> [threads] [cutoff] [batch_size]\n";
        MPI_Finalize();
        return 1;
    }

    int nt    = (argc >= 3) ? std::stoi(argv[2]) : omp_get_max_threads();
    int cut   = (argc >= 4) ? std::stoi(argv[3]) : 4;
    int batch = (argc >= 5) ? std::stoi(argv[4]) : std::max(1, nt * 2);
    omp_set_num_threads(nt);

    // ---- All ranks read the input file ----
    // This avoids the need to broadcast the (potentially large) weight matrix.
    // Since all ranks are on the same cluster filesystem, this is efficient.
    std::ifstream fin(argv[1]);
    if (!fin) {
        if (rank == 0) std::cerr << "Cannot open: " << argv[1] << "\n";
        MPI_Finalize();
        return 1;
    }

    State base{};
    fin >> base.rows >> base.cols;
    base.cost = 0; base.undecided_sum = 0;
    base.undecided_cells = base.rows * base.cols;
    base.t_count = 0; base.z_count = 0; base.next_id = 1;
    for (int i = 0; i < base.rows; i++)
        for (int j = 0; j < base.cols; j++) {
            fin >> base.weights[i][j];
            base.board[i][j] = UNDECIDED;
            base.undecided_sum += base.weights[i][j];
        }
    fin.close();

    int lb = trivial_lower_bound(base);

    // ====================================================================
    // SINGLE-PROCESS FALLBACK (no MPI parallelism, pure OpenMP)
    // ====================================================================
    if (nprocs == 1) {
        // Fall back to pure OpenMP mode (identical to original code)
        SharedBest shared;
        shared.cost = base.undecided_sum;
        shared.found_optimal = false;
        omp_init_lock(&shared.lock);
        shared.solution.cost = base.undecided_sum;

        std::cout << "Board:              " << base.rows << " x " << base.cols << "\n";
        std::cout << "Trivial lower bound: " << lb << "\n";
        std::cout << "Mode:               Pure OpenMP (" << nt << " threads)\n";
        std::cout << "Cutoff depth:       " << cut << "\n";

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<State> pool;
        generate_states(base, 0, cut, lb, pool);
        auto t1 = std::chrono::high_resolution_clock::now();
        double gen_t = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Generated tasks:     " << pool.size()
                  << "  (" << std::fixed << std::setprecision(3) << gen_t << " s)\n";

        auto t2 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < (int)pool.size(); i++) {
            if (shared.found_optimal) continue;
            State ts = pool[i];
            dfs(ts, shared, lb);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double par_t = std::chrono::duration<double>(t3 - t2).count();

        omp_destroy_lock(&shared.lock);

        std::cout << "Recursive calls:     " << g_calls << "\n";
        std::cout << "Parallel wall time:  " << std::fixed << std::setprecision(3) << par_t << " s\n";
        std::cout << "Total wall time:     " << std::fixed << std::setprecision(3) << gen_t + par_t << " s\n";
        std::cout << (shared.found_optimal ? "Result: OPTIMAL\n" : "Result: best found\n");
        print_solution(base, shared.solution);
        MPI_Finalize();
        return 0;
    }

    // ====================================================================
    // MASTER (rank 0): generate pool and distribute dynamically
    // ====================================================================
    if (rank == 0) {
        std::cout << "Board:               " << base.rows << " x " << base.cols << "\n";
        std::cout << "Trivial lower bound: " << lb << "\n";
        std::cout << "MPI processes:       " << nprocs << "\n";
        std::cout << "OpenMP threads/proc: " << nt << "\n";
        std::cout << "Cutoff depth:        " << cut << "\n";
        std::cout << "Batch size:          " << batch << "\n";

        // ---- Phase 1: generate the state pool sequentially ----
        auto tg0 = std::chrono::high_resolution_clock::now();
        std::vector<State> pool;
        generate_states(base, 0, cut, lb, pool);
        auto tg1 = std::chrono::high_resolution_clock::now();
        double gen_time = std::chrono::duration<double>(tg1 - tg0).count();
        std::cout << "Generated tasks:     " << pool.size()
                  << "  (" << std::fixed << std::setprecision(3) << gen_time << " s)\n";

        // Pack all states for MPI transfer (done once upfront to avoid
        // repeated packing; MPIState excludes constant weight data)
        std::vector<MPIState> packed(pool.size());
        for (size_t i = 0; i < pool.size(); i++)
            packed[i] = pack_state(pool[i]);
        pool.clear();                // free memory — packed version suffices

        // ---- Phase 2: dynamic Master-Slave distribution ----
        auto tp0 = std::chrono::high_resolution_clock::now();

        int global_best = base.undecided_sum;
        WorkerResult global_solution{};
        global_solution.best_cost = global_best;
        bool global_optimal = false;

        int num_workers = nprocs - 1;
        int next_idx = 0;            // next state to distribute
        int active_workers = 0;      // workers with pending work

        // --- Send initial batch to each worker ---
        for (int w = 1; w <= num_workers; w++) {
            int count = std::min(batch, (int)packed.size() - next_idx);
            if (count <= 0 || global_optimal) count = 0;

            // Header: {batch_count, current_global_best}
            int header[2] = {count, global_best};
            MPI_Send(header, 2, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);

            if (count > 0) {
                // Send the batch of packed states
                MPI_Send(&packed[next_idx], count * (int)sizeof(MPIState),
                         MPI_BYTE, w, TAG_WORK, MPI_COMM_WORLD);
                next_idx += count;
                active_workers++;
            }
            // If count == 0, worker will see header and terminate
        }

        // --- Main loop: receive results, redistribute work ---
        while (active_workers > 0) {
            // Receive result from any worker
            WorkerResult wr;
            MPI_Status status;
            MPI_Recv(&wr, sizeof(WorkerResult), MPI_BYTE,
                     MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
            int src = status.MPI_SOURCE;
            active_workers--;

            // Update global best if this worker found a better solution
            if (wr.best_cost < global_best) {
                global_best = wr.best_cost;
                global_solution = wr;
                if (wr.found_optimal) global_optimal = true;
            }

            // Send more work to this worker (or terminate it)
            int count = 0;
            if (!global_optimal && next_idx < (int)packed.size()) {
                count = std::min(batch, (int)packed.size() - next_idx);
            }

            int header[2] = {count, global_best};
            MPI_Send(header, 2, MPI_INT, src, TAG_WORK, MPI_COMM_WORLD);

            if (count > 0) {
                MPI_Send(&packed[next_idx], count * (int)sizeof(MPIState),
                         MPI_BYTE, src, TAG_WORK, MPI_COMM_WORLD);
                next_idx += count;
                active_workers++;
            }
        }

        auto tp1 = std::chrono::high_resolution_clock::now();
        double par_time = std::chrono::duration<double>(tp1 - tp0).count();

        // ---- Reduce recursive call counts from all ranks ----
        long long total_calls = 0;
        MPI_Reduce(&g_calls, &total_calls, 1, MPI_LONG_LONG, MPI_SUM,
                   0, MPI_COMM_WORLD);

        // ---- Print results ----
        std::cout << "Recursive calls:     " << total_calls << "\n";
        std::cout << "Generate time:       " << std::fixed << std::setprecision(3)
                  << gen_time << " s\n";
        std::cout << "Parallel wall time:  " << std::fixed << std::setprecision(3)
                  << par_time << " s\n";
        std::cout << "Total wall time:     " << std::fixed << std::setprecision(3)
                  << gen_time + par_time << " s\n";
        std::cout << (global_optimal ? "Result: OPTIMAL\n" : "Result: best found\n");

        // Reconstruct Best from WorkerResult for printing
        Best best_sol;
        best_sol.cost = global_solution.best_cost;
        best_sol.t_count = global_solution.t_count;
        best_sol.z_count = global_solution.z_count;
        best_sol.next_id = global_solution.next_id;
        std::memcpy(best_sol.board, global_solution.board, sizeof(best_sol.board));
        std::memcpy(best_sol.piece_type, global_solution.piece_type,
                    sizeof(best_sol.piece_type));
        print_solution(base, best_sol);
    }

    // ====================================================================
    // WORKER (rank != 0): receive batches, process with OpenMP, report back
    // ====================================================================
    else {
        while (true) {
            // ---- Receive header from master ----
            int header[2];
            MPI_Recv(header, 2, MPI_INT, 0, TAG_WORK,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int count     = header[0];
            int best_hint = header[1]; // master's current global best

            // ---- Terminate if no work ----
            if (count == 0) break;

            // ---- Receive batch of packed states ----
            std::vector<MPIState> recv_buf(count);
            MPI_Recv(recv_buf.data(), count * (int)sizeof(MPIState),
                     MPI_BYTE, 0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // ---- Unpack into full State objects ----
            std::vector<State> local_pool(count);
            for (int i = 0; i < count; i++)
                local_pool[i] = unpack_state(recv_buf[i], base);
            recv_buf.clear();

            // ---- Set up process-local SharedBest ----
            // Initialise with the master's best-known cost for tight pruning
            SharedBest shared;
            shared.cost = best_hint;
            shared.found_optimal = false;
            omp_init_lock(&shared.lock);
            shared.solution.cost = best_hint;

            // ---- Process batch with OpenMP ----
            // schedule(dynamic,1): each thread grabs the next unprocessed
            // state — handles the variance in subtree sizes from pruning
            #pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < count; i++) {
                if (shared.found_optimal) continue;
                State ts = local_pool[i];  // thread-private copy
                dfs(ts, shared, lb);
            }

            omp_destroy_lock(&shared.lock);

            // ---- Send result back to master ----
            WorkerResult wr;
            wr.best_cost = shared.cost;
            std::memcpy(wr.board, shared.solution.board, sizeof(wr.board));
            std::memcpy(wr.piece_type, shared.solution.piece_type,
                        sizeof(wr.piece_type));
            wr.t_count      = shared.solution.t_count;
            wr.z_count      = shared.solution.z_count;
            wr.next_id      = shared.solution.next_id;
            wr.found_optimal = shared.found_optimal;

            MPI_Send(&wr, sizeof(WorkerResult), MPI_BYTE,
                     0, TAG_RESULT, MPI_COMM_WORLD);
        }

        // Participate in the MPI_Reduce for call counts
        long long dummy = 0;
        MPI_Reduce(&g_calls, &dummy, 1, MPI_LONG_LONG, MPI_SUM,
                   0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}