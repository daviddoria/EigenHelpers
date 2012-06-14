/*=========================================================================
 *
 *  Copyright David Doria 2011 daviddoria@gmail.com
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

typedef std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf> > VectorOfVectors;

float SumOfRow(const Eigen::MatrixXf& m, const unsigned int rowId);

float SumOfVector(const Eigen::VectorXf& v);

float SumOfAbsoluteDifferences(const Eigen::VectorXf& a, const Eigen::VectorXf& b);

std::vector<float> EigenVectorToSTDVector(const Eigen::VectorXf& vec);

Eigen::VectorXf STDVectorToEigenVector(const std::vector<float>& vec);

void OutputVectors(const VectorOfVectors& vectors);

void OutputMatrixSize(const Eigen::MatrixXf& m);

Eigen::MatrixXf TruncateColumns(const Eigen::MatrixXf& m, const unsigned int numberOfColumnsToKeep);

/** Subtract the mean and divide by the standard deviation. */
void Standardize(EigenHelpers::VectorOfVectors& vectors);

/** Subtract the mean and divide by the standard deviation, returning the meanVector and standardDeviationVector's computed by reference. */
void Standardize(EigenHelpers::VectorOfVectors& vectors, Eigen::VectorXf& meanVector, Eigen::VectorXf& standardDeviationVector);

Eigen::VectorXf ComputeMeanVector(const EigenHelpers::VectorOfVectors& vectors);

Eigen::VectorXf ComputeMinVector(const EigenHelpers::VectorOfVectors& vectors);

Eigen::VectorXf ComputeMaxVector(const EigenHelpers::VectorOfVectors& vectors);

/** Construct the sample covariance matrix from a collection of vectors. */
Eigen::MatrixXf ConstructCovarianceMatrix(const EigenHelpers::VectorOfVectors& vectors);

/** Construct the sample covariance matrix from a feature matrix. */
Eigen::MatrixXf ConstructCovarianceMatrixFromFeatureMatrix(const Eigen::MatrixXf& featureMatrix);

/** Construct the sample covariance matrix from a collection of vectors that has already had their mean subtracted. */
Eigen::MatrixXf ConstructCovarianceMatrixZeroMean(const EigenHelpers::VectorOfVectors& vectors);

/** Construct the sample covariance matrix from a collection of vectors that has already had their mean subtracted.
  * This function constructs a matrix of the vectors and then uses a huge matrix multiplication instead of vector-at-a-time
  * constructing the covariance matrix. */
Eigen::MatrixXf ConstructCovarianceMatrixZeroMeanFast(const EigenHelpers::VectorOfVectors& vectors);

/** Project vectors into a lower dimensional space. */
EigenHelpers::VectorOfVectors DimensionalityReduction(const EigenHelpers::VectorOfVectors& vectors,
                                                      const unsigned int numberOfDimensions);

/** Project vectors into a lower dimensional space, where the covarianceMatrix has been pre-computed. */
EigenHelpers::VectorOfVectors DimensionalityReduction(const EigenHelpers::VectorOfVectors& vectors,
                                                      const Eigen::MatrixXf& covarianceMatrix,
                                                      const unsigned int numberOfDimensions);

/** Project vectors into a lower dimensional space, determining this dimensionality by keeping the number of eigenvalues necessary
  * to make their sum at least 'eigenvalueWeightToKeep' */
EigenHelpers::VectorOfVectors DimensionalityReduction(const EigenHelpers::VectorOfVectors& vectors,
                                                      const Eigen::MatrixXf& covarianceMatrix,
                                                      const float eigenvalueWeightToKeep);

/** Project a vector into a lower dimensional space, determining this dimensionality by keeping the number of eigenvalues necessary
  * to make their sum at least 'eigenvalueWeightToKeep'. 'U' is the 'U' matrix from the SVD of the covariance matrix. */
Eigen::VectorXf DimensionalityReduction(const Eigen::VectorXf& v,
                                        const Eigen::MatrixXf& U,
                                        const Eigen::VectorXf& singularValues,
                                        const float singularValueWeightToKeep);

/** Project a vector into a lower dimensional space, where the covarianceMatrix has been pre-computed.
  * 'U' is the 'U' matrix from the SVD of the covariance matrix.*/
Eigen::VectorXf DimensionalityReduction(const Eigen::VectorXf& v,
                                        const Eigen::MatrixXf& U, const unsigned int numberOfDimensions);

/** Determine how many singular values to keep to have kept a particular amount of the 'energy' of the original space */
unsigned int ComputeNumberOfSingularValuesToKeep(const Eigen::VectorXf& singularValues, const float singularValueWeightToKeep);

/** Compute the pseudo inverse of a matrix. */
Eigen::MatrixXf PseudoInverse(const Eigen::MatrixXf &m);

} // end EigenHelpers namespace

#endif
