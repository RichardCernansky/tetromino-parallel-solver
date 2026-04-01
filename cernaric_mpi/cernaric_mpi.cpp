// MPI parallelism
//Richard Cernansky, cernaric@fit.cvut.cz

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

// Constants and piece definitions
static const int MAXR = 20;
static const int MAXC = 20;
static const int UNDECIDED = 0;
static const int UNCOVERED = -1;

// T-piece: 4 rotations, each with 4 cells (row,col offsets)
const int T_VAR[4][4][2] = {
    {{0,0},{0,1},{0,2},{1,1}},
    {{0,0},{1,0},{2,0},{1,1}},
    {{0,1},{1,0},{1,1},{1,2}},
    {{1,0},{0,1},{1,1},{2,1}},
};

// Z-piece: 4 rotations (includes S-variants via flip)
const int Z_VAR[4][4][2] = {
    {{0,0},{0,1},{1,1},{1,2}},
    {{0,1},{1,0},{1,1},{2,0}},
    {{0,1},{0,2},{1,0},{1,1}},
    {{0,0},{1,0},{1,1},{2,1}},
};

// MPI message constants - flags
static const int TAG_WORK   = 10;
static const int TAG_RESULT = 20;

// Data structures
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
    int  t_count;
    int  z_count;
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
    alignas(64) int cost;
    char pad1[64];
    alignas(64) omp_lock_t lock;
    Best solution;
    bool found_optimal;
};

// Serialised state for MPI transfer (excludes weights — all ranks read file)
struct MPIState {
    int  board[MAXR][MAXC];
    char piece_type[MAXR*MAXC];
    int  cost;
    int  undecided_sum;
    int  undecided_cells;
    int  t_count;
    int  z_count;
    int  next_id;
    // dont have rows, cols
};

// Result sent from worker back to master
struct WorkerResult {
    int  best_cost;
    int  board[MAXR][MAXC];
    char piece_type[MAXR*MAXC];
    int  t_count, z_count, next_id;
    bool found_optimal;
};

// Packing  unpacking for MPI transport
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

// Unpacking for MPI transport
State unpack_state(const MPIState& ms, const State& base) {
    State s;
    std::memcpy(s.weights, base.weights, sizeof(s.weights));
    s.rows = base.rows;
    s.cols = base.cols;
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

// Core solver functions
static long long g_calls = 0;

std::pair<int,int> first_undecided(const State& s) {
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
//  Returns true  -> this branch CANNOT satisfy the parity constraint → prune.
//  Returns false -> parity is still satisfiable.
bool parity_prune(int t_count, int z_count, int undecided_cells)

{
    int diff = t_count - z_count;   // positive: more T placed
    if (diff < 0) diff = -diff;     // |diff|

    // Maximum additional pieces we could place
    int max_more = undecided_cells / 4;

    return (diff > max_more + 1);
}

int trivial_lower_bound(const State& s) {
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

// state pool generation - sequential DFS to cutoff depth
void generate_states(State s, int depth, int cutoff, int lb, std::vector<State>& pool) {
    if (depth >= cutoff) {
        pool.push_back(s);
        return;
    }
    auto [r, c] = first_undecided(s);
    if (r == -1) return;

    auto t_moves = get_placements(s, r, c, T_VAR, 'T');
    for (auto& p : t_moves) {
        apply_piece(s, p);
        generate_states(s, depth+1, cutoff, lb, pool);
        undo_piece(s, p);
    }

    auto z_moves = get_placements(s, r, c, Z_VAR, 'Z');
    for (auto& p : z_moves) {
        apply_piece(s, p);
        generate_states(s, depth+1, cutoff, lb, pool);
        undo_piece(s, p);
    }

    apply_uncover(s, r, c);
    generate_states(s, depth+1, cutoff, lb, pool);
    undo_uncover(s, r, c);
}

// DFS - OpenMP from DATA parallelized solution
void dfs(State& s, SharedBest& shared, int lb) {
    #pragma omp atomic // everyone reads
    g_calls++;

    {
        bool opt;
        #pragma omp atomic read //everyone reads
        opt = shared.found_optimal;
        if (opt) return;
    }

    int best;
    #pragma omp atomic read // everyone reads
    best = shared.cost;
    if (s.cost >= best) return;

    if (parity_prune(s.t_count, s.z_count, s.undecided_cells)) return;

    // get the undecided cell to generate possible next states
    auto [r, c] = first_undecided(s);

    if (r == -1) {
        int owned = s.cost;
        int old;
        #pragma omp atomic read
        old = shared.cost;
        if (owned < old) { //check if is smaller 
            omp_set_lock(&shared.lock);
            if (owned < shared.cost) { // still smaller?
                // write to shared, no need for atomic - lock acquired
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

    auto t_moves = get_placements(s, r, c, T_VAR, 'T');
    auto z_moves = get_placements(s, r, c, Z_VAR, 'Z');

    // calculate coverage for sort descending - largest go first, better chance
    auto cov = [&](const Placement& p) {
        int sum = 0;
        for (int i = 0; i < 4; i++) sum += s.weights[p.rows[i]][p.cols[i]];
        return sum;
    };
    std::vector<Placement> all;
    all.reserve(t_moves.size() + z_moves.size());
    for (auto& p : t_moves) all.push_back(p);
    for (auto& p : z_moves) all.push_back(p);
    std::sort(all.begin(), all.end(),
              [&](const Placement& a, const Placement& b) {
                  return cov(a) > cov(b);
              });

    // dfs on all next possible states
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

    // if optimal found return
    {
        bool opt;
        #pragma omp atomic read
        opt = shared.found_optimal;
        if (opt) return;
    }

    // dfs on possible uncover- never covered 
    int old_best;
    #pragma omp atomic read
    old_best = shared.cost;
    if (s.cost + s.weights[r][c] < old_best) {
        apply_uncover(s, r, c);
        dfs(s, shared, lb);
        undo_uncover(s, r, c);
    }
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
                : std::string(1, best.piece_type[v - 1]) + std::to_string(v);
            std::cout << std::setw(w) << cell << " ";
        }
        std::cout << "\n";
    }
}

// Main — MPI Master-Slave orchestration
int main(int argc, char* argv[]) {
    // MPI_Init is sufficient here because all MPI calls happen on the main
    // MPI never sees multiple threads calling it concurrently.
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    // initialize rank, nproc information
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 2) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0]
                      << " <file> [threads] [cutoff] [batch_size]\n";
        MPI_Finalize();
        return 1;
    }

    // number of threads 
    int nt    = (argc >= 3) ? std::stoi(argv[2]) : omp_get_max_threads();
    // cut-off depth
    int cut   = (argc >= 4) ? std::stoi(argv[3]) : 4;
    // batch size for single pass
    int batch_size = (argc >= 5) ? std::stoi(argv[4]) : std::max(1, nt * 2);
    omp_set_num_threads(nt);

    // All ranks read the input file (shared filesystem)
    std::ifstream fin(argv[1]);
    // check if can open
    if (!fin) {
        if (rank == 0) std::cerr << "Cannot open: " << argv[1] << "\n";
        MPI_Finalize();
        return 1;
    }

    // initialize base
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

    // MASTER - rank 0
    if (rank == 0) {
        std::cout << "Board:               " << base.rows << " x " << base.cols << "\n";
        std::cout << "Trivial lower bound: " << lb << "\n";
        std::cout << "MPI processes:       " << nprocs << "\n";
        std::cout << "OpenMP threads/proc: " << nt << "\n";
        std::cout << "Cutoff depth:        " << cut << "\n";
        std::cout << "Batch size:          " << batch_size << "\n";

        // start clock
        auto tg0 = std::chrono::high_resolution_clock::now();

        // generate pool of states
        std::vector<State> pool;
        generate_states(base, 0, cut, lb, pool);
        auto tg1 = std::chrono::high_resolution_clock::now();
        double gen_time = std::chrono::duration<double>(tg1 - tg0).count();
        std::cout << "Generated tasks:     " << pool.size()
                  << "  (" << std::fixed << std::setprecision(3) << gen_time << " s)\n";
        
        // pack the state pool for shipping
        std::vector<MPIState> packed(pool.size());
        for (size_t i = 0; i < pool.size(); i++)
            packed[i] = pack_state(pool[i]);
        pool.clear();

        auto tp0 = std::chrono::high_resolution_clock::now();

        // initialize global solution
        int global_best = base.undecided_sum;
        WorkerResult global_solution{};
        global_solution.best_cost = global_best;
        bool global_optimal = false;

        int num_workers = nprocs - 1; // num of workers is all nprocs without master
        int next_idx = 0;  // next state index
        int active_workers = 0; // how many workers have work

        // Send initial batch to each worker
        for (int w = 1; w <= num_workers; w++) {
            int count = std::min(batch_size, (int) packed.size() - next_idx); // calculate how many packets left
            if (count <= 0 || global_optimal) count = 0; // if count is zero or found optimal, dont send work anymore

            // create header and send to worker w 
            int header[2] = {count, global_best};
            MPI_Send(header, 2, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);

            // if has packets then send them and increase the next_idx and active workers
            if (count > 0) {
                MPI_Send(&packed[next_idx], count * (int)sizeof(MPIState),
                         MPI_BYTE, w, TAG_WORK, MPI_COMM_WORLD);
                next_idx += count;
                active_workers++;
            }
        }

        // receive results, redistribute work while active workers > 0
        while (active_workers > 0) {

            // receive the message, blocking then send anther - naive automaton 
            WorkerResult wr;
            MPI_Status status; // message metadata
            MPI_Recv(&wr, sizeof(WorkerResult), MPI_BYTE, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status); 
            int src = status.MPI_SOURCE;
            active_workers--;

            // update global best
            if (wr.best_cost < global_best) {
                global_best = wr.best_cost;
                global_solution = wr;
                if (wr.found_optimal) global_optimal = true;
            }

            // get the count of the remaining packets
            int count = 0;
            if (!global_optimal && next_idx < (int)packed.size())
                count = std::min(batch_size, (int)packed.size() - next_idx);

            // send the header first with information about count - stop condition
            int header[2] = {count, global_best};
            MPI_Send(header, 2, MPI_INT, src, TAG_WORK, MPI_COMM_WORLD);

            // if count is still > 0, send another batch, increase next_idx and active workers
            if (count > 0) {
                MPI_Send(&packed[next_idx], count * (int)sizeof(MPIState),
                         MPI_BYTE, src, TAG_WORK, MPI_COMM_WORLD);
                next_idx += count;
                active_workers++;
            }
        }

        // stop clock
        auto tp1 = std::chrono::high_resolution_clock::now();
        double par_time = std::chrono::duration<double>(tp1 - tp0).count();

        // mpi reduce for global g_calls
        long long total_calls = 0;
        // each processes private g_calls, reduced to tota_calls, one element (not array), type long, operation, root process index, group
        MPI_Reduce(&g_calls, &total_calls, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); 
        
        // time prints 
        std::cout << "Recursive calls:     " << total_calls << "\n";
        std::cout << "Generate time:       " << std::fixed << std::setprecision(3)
                  << gen_time << " s\n";
        std::cout << "Parallel wall time:  " << std::fixed << std::setprecision(3)
                  << par_time << " s\n";
        std::cout << "Total wall time:     " << std::fixed << std::setprecision(3)
                  << gen_time + par_time << " s\n";
        std::cout << (global_optimal ? "Result: OPTIMAL\n" : "Result: best found\n");
        
        // set the best solution from the global one and print it
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

    // WORKER (rank != 0)
    else {
        // for every non-master process, receive messages in infinite while
        while (true) {
            // receive message in the header
            int header[2];
            MPI_Recv(header, 2, MPI_INT, 0, TAG_WORK,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int count     = header[0];
            int best_hint = header[1];

            // stop condition when the 
            if (count == 0) break;

            // receive the message from master 
            std::vector<MPIState> recv_buf(count);
            MPI_Recv(recv_buf.data(), count * (int)sizeof(MPIState),
                     MPI_BYTE, 0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // fill the local_pool
            std::vector<State> local_pool(count);
            for (int i = 0; i < count; i++)
                local_pool[i] = unpack_state(recv_buf[i], base);
            recv_buf.clear();

            // initialize the shared best 
            SharedBest shared;
            shared.cost = best_hint;
            shared.found_optimal = false;
            omp_init_lock(&shared.lock);
            shared.solution.cost = best_hint;

            // initialize the thread parallelism with OpenMP of the worker for its local pool
            #pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < count; i++) {
                if (shared.found_optimal) continue;
                // dfs for each thread on its own copy of pooled State
                State ts = local_pool[i];
                dfs(ts, shared, lb);
            }
            omp_destroy_lock(&shared.lock);

            // fill in the WorkerResult struct and 
            WorkerResult wr;
            wr.best_cost = shared.cost;
            std::memcpy(wr.board, shared.solution.board, sizeof(wr.board));
            std::memcpy(wr.piece_type, shared.solution.piece_type,
                        sizeof(wr.piece_type));
            wr.t_count       = shared.solution.t_count;
            wr.z_count       = shared.solution.z_count;
            wr.next_id       = shared.solution.next_id;
            wr.found_optimal = shared.found_optimal;

            // send the worker result to the Master
            MPI_Send(&wr, sizeof(WorkerResult), MPI_BYTE,
                     0, TAG_RESULT, MPI_COMM_WORLD);
        }

        long long dummy = 0; // not used in non-root process
        // report private g_calls to master 
        MPI_Reduce(&g_calls, &dummy, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // finalize the program
    MPI_Finalize();
    return 0;
}