/*=========================================================================
 *
 *  Copyright David Doria 2012 daviddoria@gmail.com
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef EigenHelpers_H
#define EigenHelpers_H

// Eigen
#include <Eigen/Dense>

namespace EigenHelpers
{

/** Typedefs */
typedef std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf> > VectorOfFloatVectors;
typedef std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > VectorOfDoubleVectors;

/** Compute the sum of a row of a matrix. */
template <typename TMatrix>
float SumOfRow(const TMatrix& m, const unsigned int rowId);

template <typename TVector>
typename TVector::Scalar SumOfVector(const TVector& v);

template <typename TVector>
typename TVector::Scalar SumOfAbsoluteDifferences(const TVector& a, const TVector& b);

template <typename TVector>
std::vector<typename TVector::Scalar> EigenVectorToSTDVector(const TVector& vec);

template <typename TVector>
TVector STDVectorToEigenVector(const std::vector<typename TVector::Scalar>& vec);

template <typename TVectorOfVectors>
void OutputVectors(const TVectorOfVectors& vectors);

template <typename TVector>
void OutputHorizontal(const TVector& v);

template <typename TVector>
void OutputHorizontal(const std::string& name, const TVector& v);

template <typename TMatrix>
void OutputMatrixSize(const TMatrix& m);

/** Keep only the first numberOfColumnsToKeep columns. That is, an m-x-n matrix becomes m-x-numberOfColumnsToKeep. */
template <typename TMatrix>
TMatrix TruncateColumns(const TMatrix& m, const unsigned int numberOfColumnsToKeep);

/** Keep only the first numberOfRowsToKeep rows. That is, an m-x-n matrix becomes numberOfRowsToKeep-x-n. */
template <typename TMatrix>
TMatrix TruncateRows(const TMatrix& m, const unsigned int numberOfRowsToKeep);

/** Subtract the mean and divide by the standard deviation. */
template <typename TVectorOfVectors>
void Standardize(TVectorOfVectors& vectors);

/** Subtract the mean and divide by the standard deviation,
  * returning the meanVector and standardDeviationVector's computed by reference. */
template <typename TVectorOfVectors>
void Standardize(TVectorOfVectors& vectors, typename TVectorOfVectors::value_type& meanVector,
                 typename TVectorOfVectors::value_type& standardDeviationVector);

template <typename TVectorOfVectors>
typename TVectorOfVectors::value_type ComputeMeanVector(const TVectorOfVectors& vectors);

template <typename TVectorOfVectors>
typename TVectorOfVectors::value_type ComputeMinVector(const TVectorOfVectors& vectors);

template <typename TVectorOfVectors>
typename TVectorOfVectors::value_type ComputeMaxVector(const TVectorOfVectors& vectors);

/** Construct the sample covariance matrix from a collection of vectors. Note that the eigenvalues of the covariance matrix are equal to
  * the singular values of the matrix itself. */
template <typename TMatrix, typename TVectorOfVectors>
TMatrix ConstructCovarianceMatrix(const TVectorOfVectors& vectors);

/** Construct the sample covariance matrix from a feature matrix. */
template <typename TMatrix>
TMatrix ConstructCovarianceMatrixFromFeatureMatrix(const TMatrix& featureMatrix);

/** Construct the sample covariance matrix from a collection of vectors that has already had their mean subtracted. */
template <typename TMatrix, typename TVectorOfVectors>
TMatrix ConstructCovarianceMatrixZeroMean(const TVectorOfVectors& vectors);

/** Construct the sample covariance matrix from a collection of vectors that has already had their mean subtracted.
  * This function constructs a matrix of the vectors and then uses a huge matrix multiplication instead of vector-at-a-time
  * constructing the covariance matrix. */
template <typename TMatrix, typename TVectorOfVectors>
TMatrix ConstructCovarianceMatrixZeroMeanFast(const TVectorOfVectors& vectors);

/** Project vectors into a lower dimensional space. */
template <typename TVectorOfVectors>
TVectorOfVectors DimensionalityReduction(const TVectorOfVectors& vectors,
                                         const unsigned int numberOfDimensions);

/** Project vectors into a lower dimensional space, where the covarianceMatrix has been pre-computed.
  * We cannot use the typedef's for the VectorOfVectors here, because another function would have an identical signature.
  * That is, DimensionalityReduction(TVectorOfVectors, TMatrix, uint), would have exactly the same signature as
  * DimensionalityReduction(TVector, TMatrix, uint). */
template <typename TMatrix, typename TVector>
std::vector<TVector, Eigen::aligned_allocator<TVector> >
DimensionalityReduction(const std::vector<TVector, Eigen::aligned_allocator<TVector> >& vectors,
                        const TMatrix& covarianceMatrix,
                        const unsigned int numberOfDimensions);

/** Project vectors into a lower dimensional space, determining this dimensionality by keeping the number of eigenvalues necessary
  * to make their sum at least 'eigenvalueWeightToKeep' */
template <typename TMatrix, typename TVectorOfVectors>
TVectorOfVectors DimensionalityReduction(const TVectorOfVectors& vectors,
                                         const TMatrix& covarianceMatrix,
                                         const float eigenvalueWeightToKeep);

/** Project a vector into a lower dimensional space, determining this dimensionality by keeping the number of eigenvalues necessary
  * to make their sum at least 'eigenvalueWeightToKeep'. 'U' is the 'U' matrix from the SVD of the covariance matrix. */
template <typename TMatrix, typename TVector>
TVector DimensionalityReduction(const TVector& v, const TMatrix& U,
                                const TVector& singularValues,
                                const float singularValueWeightToKeep);

/** Project a vector into a lower dimensional space, where the covarianceMatrix has been pre-computed.
  * 'U' is the 'U' matrix from the SVD of the covariance matrix.*/
template <typename TMatrix, typename TVector>
TVector DimensionalityReduction(const TVector& v, const TMatrix& U, const unsigned int numberOfDimensions);

/** Determine how many singular values to keep to have kept a particular amount of the 'energy' of the original space */
template <typename TVector>
unsigned int ComputeNumberOfSingularValuesToKeep(const TVector& singularValues, const float singularValueWeightToKeep);

/** Compute the pseudo inverse of a matrix. */
template <typename TMatrix>
TMatrix PseudoInverse(const TMatrix &m);

/** Scale the values of a vector. */
template <typename TVector>
TVector ScaleVector(const TVector& v, const typename TVector::Scalar& lower,
                    const typename TVector::Scalar& upper);

/** Construct a random unit vector with dimension 'dim'. */
template <typename TVector>
TVector RandomUnitVector(const unsigned int dim);

/** Get the min and max corner of the bounding box of the data. Both are returned by reference. */
template <typename TPoint>
void GetBoundingBox(const std::vector<TPoint, Eigen::aligned_allocator<TPoint> >& data, TPoint& minCorner, TPoint& maxCorner);

} // end EigenHelpers namespace

#include "EigenHelpers.hpp"

#endif
