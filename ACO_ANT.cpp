#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <SFML/Graphics.hpp> 

struct Vector2 {
    float x, y;

    Vector2() : x(0), y(0) {}
    Vector2(float x, float y) : x(x), y(y) {}

    Vector2 operator-(const Vector2& other) const {
        return Vector2(x - other.x, y - other.y);
    }

    Vector2 operator+(const Vector2& other) const {
        return Vector2(x + other.x, y + other.y);
    }

    Vector2 operator*(float scalar) const {
        return Vector2(x * scalar, y * scalar);
    }

    Vector2& operator+=(const Vector2& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    float length() const {
        return std::sqrt(x * x + y * y);
    }

    float lengthSquared() const {
        return x * x + y * y;
    }

    Vector2 normalized() const {
        float len = length();
        return len > 0 ? Vector2(x / len, y / len) : Vector2(0, 0);
    }

    Vector2 rotated(float angle) const {
        float cs = std::cos(angle), sn = std::sin(angle);
        return Vector2(cs * x - sn * y, sn * x + cs * y);
    }
};


class ACO {
public:
    struct Node {
        Vector2 position;
    };

    struct Edge {
        Vector2 position;
        float rotation;
        Vector2 scale;
        Vector2 size;
    };

    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> distances;

    std::default_random_engine rng;
    std::uniform_real_distribution<float> dist;

    int num_nodes = 20;
    float node_radius = 50;
    float edge_width = 100;
    float edge_scale = 0.1;
    float best_len = std::numeric_limits<float>::max();
    float alpha = 1.0;
    float beta = 2.0;
    bool running = false;

    ACO() : dist(0.0, 1.0) {
        rng.seed(std::random_device()());
    }

    void generate(int screen_width, int screen_height) {
        best_len = std::numeric_limits<float>::max();
        nodes.clear();
        edges.clear();

        int L = screen_width - 2 * node_radius;
        int W = screen_height - 2 * node_radius;

        for (int _ = 0; _ < num_nodes; ++_) {
            addNode(rng() % L + node_radius, rng() % W + node_radius);
        }

        spreadNodes(40);
    }

    void addNode(float x, float y) {
        nodes.push_back(Node{ Vector2(x, y) });
    }

    void addEdge(int i1, int i2) {
        Edge edge;
        edge.position = nodes[i1].position;
        Vector2 displacement = nodes[i2].position - nodes[i1].position;
        edge.rotation = std::atan2(displacement.y, displacement.x);
        float scale = weights[i1][i2] * edge_scale;
        edge.scale = Vector2(scale, scale);
        edges.push_back(edge);
    }

    void addWeights() {
        weights.assign(num_nodes, std::vector<float>(num_nodes, 1.0f));
        distances.assign(num_nodes, std::vector<float>(num_nodes, 0.0f));

        for (int i = 0; i < num_nodes; i++) {
            for (int j = 0; j < num_nodes; j++) {
                if (i == j) continue;
                distances[i][j] = (nodes[i].position - nodes[j].position).length();
            }
        }
    }

    void spreadNodes(int iterations) {
        for (int _ = 0; _ < iterations; ++_) {
            Vector2 center(0, 0);
            for (auto& node : nodes) {
                center += node.position;
            }
            center = center * (1.0f / num_nodes);

            std::vector<Vector2> new_positions(num_nodes);
            for (int i = 0; i < num_nodes; ++i) {
                new_positions[i] = nodes[i].position;
                for (int j = 0; j < num_nodes; ++j) {
                    if (i == j) continue;
                    new_positions[i] += force(nodes[i].position, nodes[j].position, false);
                }
                new_positions[i] += force(nodes[i].position, center, true);
            }

            for (int i = 0; i < num_nodes; ++i) {
                nodes[i].position = new_positions[i];
            }
        }

        addWeights();
    }

    Vector2 force(const Vector2& me, const Vector2& other, bool attractive) {
        float ka = 0.1f * num_nodes;
        float kr = -10000;
        float eps = 0.001f;
        float dist_squared = (me - other).lengthSquared();
        float factor = (dist_squared > eps) ? (1.0f / dist_squared) : (1.0f / eps);
        factor = attractive ? ka : factor * kr;

        return (other - me).normalized() * factor;
    }

    std::pair<std::vector<int>, float> getRandomPath(int start_idx) {
        std::vector<int> path;
        path.push_back(start_idx);
        int current_idx = start_idx;
        float total_distance = 0.0f;

        while (path.size() < num_nodes) {
            float total_prob = 0.0f;
            std::vector<int> candidates;

            for (int i = 0; i < num_nodes; ++i) {
                if (std::find(path.begin(), path.end(), i) == path.end()) {
                    total_prob += getTransitionProbability(current_idx, i);
                    candidates.push_back(i);
                }
            }

            float r = dist(rng) * total_prob;
            float cumulative_prob = 0.0f;
            for (int candidate : candidates) {
                cumulative_prob += getTransitionProbability(current_idx, candidate);
                if (r <= cumulative_prob) {
                    total_distance += distances[current_idx][candidate];
                    current_idx = candidate;
                    path.push_back(candidate);
                    break;
                }
            }
        }

        total_distance += distances[current_idx][start_idx];
        return { path, total_distance };
    }

    float getTransitionProbability(int idx1, int idx2) {
        return std::pow(weights[idx1][idx2], alpha) * std::pow(1.0f / distances[idx1][idx2], beta);
    }

    void runACOBatch(int batch_size) {
        running = false;

        for (auto& row : weights) {
            for (auto& weight : row) {
                weight *= 0.999f; // Evaporation
            }
        }

        auto new_weights = weights;

        for (int _ = 0; _ < batch_size; ++_) {
            auto [path, length] = getRandomPath(rng() % num_nodes);
            best_len = std::min(best_len, length);

            float diff = length - best_len + 0.05f;
            float pheromone = 0.01f / diff;

            for (size_t i = 0; i < path.size(); ++i) {
                int idx1 = path[i % num_nodes];
                int idx2 = path[(i + 1) % num_nodes];
                new_weights[idx1][idx2] += pheromone;
                new_weights[idx2][idx1] += pheromone;
            }
        }

        for (int i = 0; i < num_nodes; ++i) {
            float sum_weights = 0.0f;
            for (int j = 0; j < num_nodes; ++j) {
                if (i != j) {
                    sum_weights += new_weights[i][j];
                }
            }
            for (int j = 0; j < num_nodes; ++j) {
                weights[i][j] = 2.0f * new_weights[i][j] / sum_weights;
            }
        }
    }
};

int main() {
   
    ACO aco;
    aco.generate(800, 600); 
    aco.runACOBatch(20);

    return 0;
}


