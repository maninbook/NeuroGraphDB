/* 파이썬 바인딩. 실행은 파이썬에서 하되 색인·검색·채점은 전부 C++에서 돈다. */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "bm25.h"
#include "dense.h"
#include "graph.h"

namespace py = pybind11;

PYBIND11_MODULE(_ngdb_core, m) {
    m.doc() = "NeuroGraphDB baseline retrieval core (C++)";

    py::class_<BM25>(m, "BM25")
        .def(py::init<float,float>(), py::arg("k1") = 1.2f, py::arg("b") = 0.75f)
        .def("add", &BM25::add, py::arg("text"))
        .def("finalize", &BM25::finalize)
        .def("search", [](const BM25& self, const std::string& q, int k) {
            std::vector<std::pair<int,float>> out;
            for (auto& r : self.search(q, k)) out.emplace_back(r.doc, r.score);
            return out;
        }, py::arg("query"), py::arg("k") = 10)
        .def_property_readonly("size", &BM25::size);

    py::class_<Graph>(m, "Graph")
        .def(py::init<int32_t>(), py::arg("n_nodes"))
        .def("add_edge", &Graph::add_edge, py::arg("src"), py::arg("dst"), py::arg("w") = 1.0f)
        .def("add_edges", [](Graph& self, const std::vector<std::tuple<int,int,float>>& es) {
            for (auto& [s, d, w] : es) self.add_edge(s, d, w);
        }, py::arg("edges"))
        /* 명제 엣지 — (src, dst, w, 술어타입, 극성) */
        .def("add_edges_typed", [](Graph& self,
                const std::vector<std::tuple<int,int,float,int,int>>& es) {
            for (auto& [s, d, w, t, p] : es)
                self.add_edge_typed(s, d, w, (int16_t)t, (int8_t)p);
        }, py::arg("edges"))
        .def("spread", [](const Graph& self, const std::vector<int32_t>& seeds,
                          const std::vector<float>& acts, int max_depth,
                          float depth_decay, float min_act, int top_k,
                          const std::vector<int16_t>& query_types,
                          float alpha, float beta) {
            auto r = self.spread(seeds, acts, max_depth, depth_decay, min_act,
                                 query_types, alpha, beta);
            if (top_k > 0 && (int)r.size() > top_k) r.resize(top_k);
            std::vector<std::pair<int,float>> out;
            out.reserve(r.size());
            for (auto& x : r) out.emplace_back(x.doc, x.score);
            return out;
        }, py::arg("seeds"), py::arg("seed_acts"), py::arg("max_depth") = 3,
           py::arg("depth_decay") = 0.65f, py::arg("min_act") = 0.02f,
           py::arg("top_k") = 0,
           /* 기본값이 "제약 없음"이라 기존 호출부는 한 글자도 안 바꿔도 된다 */
           py::arg("query_types") = std::vector<int16_t>{},
           py::arg("alpha") = 1.0f, py::arg("beta") = 1.0f)
        .def("reinforce", &Graph::reinforce, py::arg("coactive"),
             py::arg("delta") = 0.03f, py::arg("w_max") = 1.0f)
        .def("decay", &Graph::decay, py::arg("lambda_") = 0.02f)
        .def("edge_weight", &Graph::edge_weight, py::arg("src"), py::arg("dst"))
        .def_property_readonly("n_nodes", &Graph::n_nodes)
        .def_property_readonly("n_edges", &Graph::n_edges);

    py::class_<DenseIndex>(m, "DenseIndex")
        .def(py::init<int>(), py::arg("dim"))
        .def("add", [](DenseIndex& self, py::array_t<float, py::array::c_style | py::array::forcecast> v) {
            if (v.ndim() != 1 || v.shape(0) != self.dim())
                throw std::runtime_error("벡터 차원이 색인과 다르다");
            return self.add(v.data());
        }, py::arg("vec"))
        .def("add_batch", [](DenseIndex& self, py::array_t<float, py::array::c_style | py::array::forcecast> m) {
            if (m.ndim() != 2 || m.shape(1) != self.dim())
                throw std::runtime_error("배치 shape이 (n, dim)이 아니다");
            std::vector<int32_t> ids;
            ids.reserve(m.shape(0));
            for (py::ssize_t i = 0; i < m.shape(0); i++)
                ids.push_back(self.add(m.data() + i * self.dim()));
            return ids;
        }, py::arg("mat"))
        .def("search", [](const DenseIndex& self, py::array_t<float, py::array::c_style | py::array::forcecast> q, int k) {
            if (q.ndim() != 1 || q.shape(0) != self.dim())
                throw std::runtime_error("질의 벡터 차원이 색인과 다르다");
            std::vector<std::pair<int,float>> out;
            for (auto& r : self.search(q.data(), k)) out.emplace_back(r.doc, r.score);
            return out;
        }, py::arg("vec"), py::arg("k") = 10)
        .def_property_readonly("size", &DenseIndex::size)
        .def_property_readonly("dim", &DenseIndex::dim);
}
