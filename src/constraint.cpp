/**
 * $Id: $
 */

#ifndef CONSTRAINT_HPP_
#define CONSTRAINT_HPP_

#include "pcalg/constraint.hpp"
#include "pcalg/gies_debug.hpp"

#include <omp.h>
#include <algorithm>
#include <utility>
#include <iterator>
#include <limits>
#include <chrono>
#include <math.h>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/dynamic_bitset.hpp>

static double igf_own(double S, double Z)
{
    if(Z < 0.0){
      return 0.0;
    }
    double Sc = (1.0 / S);
    Sc *= pow(Z, S);
    Sc *= exp(-Z);

    double Sum = 1.0;
    double Nom = 1.0;
    double Denom = 1.0;

    for(int I = 0; I < 200; I++){
      Nom *= Z;
      S++;
      Denom *= S;
      Sum += (Nom / Denom);
    }

    return Sum * Sc;
}

static double pchisq_own(double chi_squared, int dof){
  if(chi_squared < 0 || dof < 1)
    {
      // TODO check if we have to return 1.0 for lower.tail = FALSE
        return 1.0;
    }
    double K = ((double)dof) * 0.5;
    double X = chi_squared * 0.5;
    if(dof == 2){
      // TODO check if we have to return 1 - exp(-1.0 * X) for lower.tail = FALSE
      return (exp(-1.0 * X));
    }

    double PValue = igf_own(K, X);
    if(PValue <= 1e-8){
      return 1 - 1e-14;
    }
    if(isnan(PValue) || isinf(PValue)){
        return (1e-14);
    }

    PValue /= tgamma(K);

    return (1.0 - PValue);
}

double IndepTestRFunction::test(uint u, uint v, std::vector<uint> S) const
{
	// Adapt indices to R convention
	std::vector<uint> shiftS;
	shiftS.reserve(S.size());
	std::vector<uint>::iterator vi;
	for (vi = S.begin(); vi != S.end(); ++vi)
		shiftS.push_back(*vi + 1);

	// Call R function to perform independence test
	return Rcpp::as<double>(_testFunction(u + 1, v + 1, shiftS, *_suffStat));
}

double IndepTestDisci::test(uint u, uint v, std::vector<uint> S) const
{
	int dof = 0;
	double chi_squared = 0.0;
	double observation_count = (double) _observations.n_rows;
	if (S.empty()){
		int u_cat_count = _nlev(u);
		int v_cat_count = _nlev(v);
		int contingency[u_cat_count * v_cat_count] = {0};
		int marginal_sums_u[u_cat_count] = {0};
		int marginal_sums_v[v_cat_count] = {0};
		for(int i = 0; i < _observations.n_rows; ++i){
			contingency[_observations(i, u) * v_cat_count + _observations(i, v)] += 1;
		}
		for(int i = 0; i < u_cat_count; ++i){
			for(int j = 0; j < v_cat_count; ++j){
				marginal_sums_u[i] += contingency[(i*v_cat_count)+j];
				marginal_sums_v[j] += contingency[(i*v_cat_count)+j];
			}
		}
		for(int i = 0; i < u_cat_count; ++i){
			for(int j = 0; j < v_cat_count; ++j){
				double expected = marginal_sums_u[i] * (double)marginal_sums_v[j] / _observations.n_rows;
				if(expected != 0){
					chi_squared += (contingency[i * v_cat_count + j] - expected) * (contingency[i * v_cat_count + j] - expected) / expected;
				}
			}
		}
		dof = (u_cat_count - 1) * (v_cat_count - 1);

	}else {
		int u_cat_count = _nlev(u);
		int v_cat_count = _nlev(v);
		int s_cat_count = 1;
		for ( auto &i : S ) {
			s_cat_count *= _nlev(i);
		}
		int contingency[u_cat_count * v_cat_count * s_cat_count] = {0};
		int indep1_sums[u_cat_count * s_cat_count] = {0};
		int indep2_sums[v_cat_count * s_cat_count] = {0};
		int total_sums[s_cat_count] = {0};

		for (int i = 0; i < _observations.n_rows; ++i){
			int multiplier = 1;
			int cond_index = 0;
			for (auto &j : S) {
				cond_index += _observations(i, j) * multiplier;
				multiplier *= _nlev(j);
			}
			contingency[cond_index * u_cat_count * v_cat_count + _observations(i, u) * v_cat_count + _observations(i, v)]++;
		}

		for(int i = 0; i < s_cat_count; ++i){
			for (int j = 0; j < u_cat_count; ++j){
				for (int k = 0; k < v_cat_count; ++k){
					indep1_sums[i* u_cat_count + j] += contingency[i* u_cat_count * v_cat_count + j * v_cat_count + k];
					indep2_sums[i* v_cat_count + k] += contingency[i* u_cat_count * v_cat_count + j * v_cat_count + k];
					total_sums[i] += contingency[i* u_cat_count * v_cat_count + j * v_cat_count + k];
				}
			}
		}

		double expected = 0;
		for(int i = 0; i < s_cat_count; ++i){
			if (total_sums[i] == 0){
				continue;
			}

			for (int k = 0; k < v_cat_count; ++k){
				for (int j = 0; j < u_cat_count; ++j){
					expected = indep1_sums[i*u_cat_count + j] * (double) indep2_sums[i*v_cat_count +k]  / total_sums[i];
					if(expected != 0.0){
						chi_squared += (contingency[i* u_cat_count * v_cat_count + j * v_cat_count + k] - expected) *
						( contingency[i* u_cat_count * v_cat_count + j * v_cat_count + k] - expected) / expected;
					}
				}
			}
		}
		dof = (u_cat_count - 1) * (v_cat_count - 1) * s_cat_count;
	}
	//return pchisq_own(chi_squared, dof);
	//dout.level(1) << "u: " << u << " v: " << v << " dof: " << dof << " chi_squared: " << chi_squared << " pchisq: " << R::pchisq(chi_squared, dof, FALSE, FALSE) << std::endl; // debug own
	return R::pchisq(chi_squared, dof, FALSE, FALSE);
}

double IndepTestGauss::test(uint u, uint v, std::vector<uint> S) const
{
	// Return NaN if any of the correlation coefficients needed for calculation is NaN
	arma::mat C_sub;
	arma::uvec ind(S.size() + 2);
	ind(0) = u;
	ind(1) = v;
	uint i, j;
	for (i = 0; i < S.size(); ++i) ind(i + 2) = S[i];
	C_sub = _correlation.submat(ind, ind);
	for (i = 0; i < C_sub.n_rows; ++i)
		for (j = 0; j < C_sub.n_cols; ++j)
			if ((boost::math::isnan)(C_sub(i, j)))
				return std::numeric_limits<double>::quiet_NaN();

	// Calculate (absolute value of) z statistic
	#define CUT_THR 0.9999999
	double r, absz;
	//dout.level(3) << " Performing independence test for conditioning set of size " << S.size() << std::endl;
	if (S.empty())
		r = _correlation(u, v);
	else if (S.size() == 1)
		r = (C_sub(0, 1) - C_sub(0, 2) * C_sub(1, 2))/sqrt((1 - C_sub(1, 2)*C_sub(1, 2)) * (1 - C_sub(0, 2)*C_sub(0, 2)));
	else {
		arma::mat PM;
		pinv(PM, C_sub);
		// TODO include error handling
		r = - PM(0, 1)/sqrt(PM(0, 0) * PM(1, 1));
	}
	// Absolute value of r, respect cut threshold
	r = std::min(CUT_THR, std::abs(r));

	// Absolute value of z statistic
	// Note: log1p for more numerical stability, see "Aaux.R"; log1p is also available in
	// header <cmath>, but probably only on quite up to date headers (C++11)?
	absz = sqrt(_sampleSize - S.size() - 3.0) * 0.5 * boost::math::log1p(2*r/(1 - r));

	// Calculate p-value to z statistic (based on standard normal distribution)
	boost::math::normal distN;
	return (2*boost::math::cdf(boost::math::complement(distN, absz)));
}

void Skeleton::addFixedEdge(const uint a, const uint b)
{
	boost::add_edge(a, b, _fixedEdges);
	addEdge(a, b);
}

bool Skeleton::isFixed(const uint a, const uint b) const
{
	bool result;
	boost::tie(boost::tuples::ignore, result) = boost::edge(a, b, _fixedEdges);
	return result;
}

void Skeleton::removeEdge(const uint a, const uint b)
{
	if (!isFixed(a, b))
		boost::remove_edge(a, b, _graph);
}

bool Skeleton::hasEdge(const uint a, const uint b) const
{
	bool result;
	boost::tie(boost::tuples::ignore, result) = boost::edge(a, b, _graph);
	return result;
}


std::set<uint> Skeleton::getNeighbors(const uint vertex) const
{
	std::set<uint> result;
	UndirOutEdgeIter outIter, outLast;

	for (boost::tie(outIter, outLast) = boost::out_edges(vertex, _graph); outIter != outLast; outIter++)
			result.insert(boost::target(*outIter, _graph));

	return result;
}

Rcpp::LogicalMatrix Skeleton::getAdjacencyMatrix()
{
	Rcpp::LogicalMatrix result(getVertexCount(), getVertexCount());
	UndirEdgeIter ei, eiLast;
	for (boost::tie(ei, eiLast) = boost::edges(_graph); ei != eiLast; ei++) {
		dout.level(3) << "  Edge {" << boost::source(*ei, _graph) <<
				", " << boost::target(*ei, _graph) << "}\n";
		result(boost::source(*ei, _graph), boost::target(*ei, _graph)) = true;
		result(boost::target(*ei, _graph), boost::source(*ei, _graph)) = true;
	}

	return result;
}

void Skeleton::fitCondInd(
		const double alpha,
		Rcpp::NumericMatrix& pMax,
		SepSets& sepSet,
		std::vector<int>& edgeTests,
		int maxCondSize,
		const bool NAdelete) {
	if (maxCondSize < 0)
		maxCondSize = getVertexCount();

	dout.level(2) << "Significance level " << alpha << std::endl;
	dout.level(2) << "Maximum order: " << maxCondSize << std::endl;
	bool found = true;

	UndirEdgeIter ei, eiLast;

	int threads = 1;
#pragma omp parallel
	{
		threads = omp_get_num_threads();
	}
	dout.level(1) << "Number of threads used in level larger zero " << threads << std::endl;
	// edgeTests lists the number of edge tests that have already been done; its size
	// corresponds to the size of conditioning sets that have already been checked
	// TODO: improve handling of check_interrupt, see e.g.
	// https://github.com/jjallaire/Rcpp/blob/master/inst/examples/OpenMP/piWithInterrupts.cpp
	for (uint condSize = edgeTests.size();
			!check_interrupt() && found && (int)condSize <= maxCondSize;
			++condSize) {
		auto start_l = std::chrono::system_clock::now();
		dout.level(1) << "Order = " << condSize << "; remaining edges: " << getEdgeCount() << std::endl;

		// Make a list of edges in the graph; this is needed for OpenMP
		std::vector<uint> u, v;
		u.reserve(getEdgeCount());
		v.reserve(getEdgeCount());
		for (boost::tie(ei, eiLast) = boost::edges(_graph); ei != eiLast && !check_interrupt(); ei++) {
			uint node1 = boost::source(*ei, _graph);
			uint node2 = boost::target(*ei, _graph);
			if (node1 > node2)
				std::swap(node1, node2);
			if (std::max(getDegree(node1), getDegree(node2)) > condSize && !isFixed(node1, node2)) {
				u.push_back(node1);
				v.push_back(node2);
			}
		}
		//boost::dynamic_bitset<> deleteEdges(u.size());
		std::vector<uint> deleteEdges(u.size(), 0);
		arma::ivec localEdgeTests(u.size(), arma::fill::zeros);

		// There is a conditioning set of size "condSize" if u is not empty
		found = u.size() > 0;

		// Iterate over all edges in the graph
		size_t uSize = u.size();
		#pragma omp parallel for
		for (std::size_t l = 0; l < uSize; l++) {
			bool edgeDone = false;

			int k;
			UndirOutEdgeIter outIter, outLast;
			std::vector<uint> condSet(condSize);
			std::vector<std::vector<uint>::iterator> si(condSize);

			// Check neighborhood of u
			if (getDegree(u[l]) > condSize) {
				// Get neighbors of u (except v)
				std::vector<uint> neighbors(0);
				neighbors.reserve(getDegree(u[l]) - 1);
				for (boost::tie(outIter, outLast) = boost::out_edges(u[l], _graph); outIter != outLast; outIter++)
					if (boost::target(*outIter, _graph) != v[l])
						neighbors.push_back(boost::target(*outIter, _graph));

				// Initialize first conditioning set
				for (std::size_t i = 0; i < condSize; ++i)
					si[i] = neighbors.begin() + i;

				// Iterate over conditioning sets
				do {
					for (std::size_t i = 0; i < condSize; ++i)
						condSet[i] = *(si[i]);

					// Test of u and v are conditionally independent given condSet
					double pval = _indepTest->test(u[l], v[l], condSet);
					localEdgeTests(l)++;
					//dout.level(1) << "  x = " << u[l] << ", y = " << v[l] << ", S = " <<
					//		condSet << " : pval = " << pval << std::endl;
					if ((boost::math::isnan)(pval))
						pval = (NAdelete ? 1. : 0.);
					if (pval > pMax(u[l], v[l]))
						pMax(u[l], v[l]) = pval;
					if (pval >= alpha) {
						//deleteEdges.set(l);
						deleteEdges[l] = 1;
						// arma::ivec condSetR(condSet.size());
						sepSet[v[l]][u[l]].set_size(condSet.size());
						for (std::size_t j = 0; j < condSet.size(); ++j)
							sepSet[v[l]][u[l]][j] = condSet[j] + 1;
						edgeDone = true;
						break; // Leave do-while-loop
					}

					// Proceed to next conditioning set
					for (k = condSize - 1;
							k >= 0 && si[k] == neighbors.begin() + (neighbors.size() - condSize + k);
							--k);
					if (k >= 0) {
						si[k]++;
						for (k++; k < (int)condSize; ++k)
							si[k] = si[k - 1] + 1;
					}
				} while(k >= 0);
			} // IF getDegree(u[l])

			// Check neighborhood of v
			if (!edgeDone && getDegree(v[l]) > condSize) {
				// Get neighbors of u (except v); common neighbors of u and v are listed in the end
				std::vector<uint> neighbors(0);
				std::vector<uint> commNeighbors(0);
				neighbors.reserve(getDegree(v[l]) - 1);
				commNeighbors.reserve(getDegree(v[l]) - 1);
				uint a;
				for (boost::tie(outIter, outLast) = boost::out_edges(v[l], _graph);
						outIter != outLast; outIter++) {
					a = boost::target(*outIter, _graph);
					if (a != u[l]) {
						if (hasEdge(u[l], a))
							commNeighbors.push_back(a);
						else
							neighbors.push_back(a);
					}
				}

				// m: number of neighbors of v that are not neighbors of u
				uint m = neighbors.size();
				neighbors.insert(neighbors.end(), commNeighbors.begin(), commNeighbors.end());
				//dout.level(2) << "  v: " << v << "; neighbors: " << neighbors << " (m = " << m << ")\n";

				// If all neighbors of v are also adjacent to u: already checked all conditioning sets
				if (m > 0) {
					// Initialize first conditioning set
					for (std::size_t i = 0; i < condSize; ++i)
						si[i] = neighbors.begin() + i;

					// Iterate over conditioning sets
					do {
						for (std::size_t i = 0; i < condSize; ++i)
							condSet[i] = *(si[i]);

						// Test of u and v are conditionally independent given condSet
						double pval = _indepTest->test(v[l], u[l], condSet);
						localEdgeTests(l)++;
				//		dout.level(1) << "  x = " << v[l] << ", y = " << u[l] << ", S = " <<
				//				condSet << " : pval = " << pval << std::endl;
						if ((boost::math::isnan)(pval))
							pval = (NAdelete ? 1. : 0.);
						if (pval > pMax(u[l], v[l]))
							pMax(u[l], v[l]) = pval;
						if (pval >= alpha) {
							//deleteEdges.set(l);
							deleteEdges[l] = 1;
							// arma::ivec condSetR(condSet.size());
							sepSet[v[l]][u[l]].set_size(condSet.size());
							for (std::size_t j = 0; j < condSet.size(); ++j)
								sepSet[v[l]][u[l]][j] = condSet[j] + 1;
							edgeDone = true;
							break; // Leave do-while-loop
						}

						// Proceed to next conditioning set
						for (k = condSize - 1;
								k >= 0 && si[k] == neighbors.begin() + (neighbors.size() - condSize + k);
								--k);
						// Make sure first element does not belong to neighborhood of u: otherwise
						// we would redo a test already performed
						if (k == 0 && si[0] == neighbors.begin() + (m - 1))
							k = -1;
						if (k >= 0) {
							si[k]++;
							for (k++; k < (int)condSize; ++k)
								si[k] = si[k - 1] + 1;
						}
					} while(k >= 0);
				} // IF m
			} // IF getDegree(v[l])
		} // FOR l

		// Delete edges marked for deletion
		//for (std::size_t l = deleteEdges.find_first(); l < deleteEdges.size(); l = deleteEdges.find_next(l))
		//	removeEdge(u[l], v[l]);
		for (std::size_t l = 0; l < deleteEdges.size(); ++l){
			if(deleteEdges[l] == 1){
				removeEdge(u[l], v[l]);
			}
		}
		// Calculate total number of edge tests
		if (found)
			edgeTests.push_back(arma::accu(localEdgeTests));
		auto duration_l = std::chrono::duration_cast<std::chrono::microseconds>
			(std::chrono::system_clock::now() - start_l).count();
		dout.level(1) << "Exec Level "<< condSize << " in microseconds: " << duration_l << std::endl;
	} // FOR condSize
}

#endif /* CONSTRAINT_HPP_ */
