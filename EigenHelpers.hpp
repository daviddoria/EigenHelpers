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

#ifndef EigenHelpers_HPP
#define EigenHelpers_HPP

#include "EigenHelpers.h"

#include <iostream>
#include <stdexcept>

namespace EigenHelpers
{

template <typename TMatrix>
float SumOfRow(const TMatrix& m, const unsigned int rowId)
{
  return SumOfVector(m.row(rowId));
}

template <typename TVector>
typename TVector::Scalar SumOfVector(const TVector& v)
{
  float sum = 0.0f;
  for(unsigned int i = 0; i < static_cast<unsigned int>(v.size()); ++i)
  {
    sum += v[i];
  }
  return sum;
}

template <typename TVector>
typename TVector::Scalar SumOfAbsoluteDifferences(const TVector& a, const TVector& b)
{
  if(a.size() != b.size())
  {
    throw std::runtime_error("SumOfAbsoluteDifferences: vectors must be the same size!");
  }

  float total = 0.0f;
  for(unsigned int i = 0; i < static_cast<unsigned int>(a.size()); ++i)
  {
    total += fabs(a[i] - b[i]);
  }

  return total;
}

template <typename TVector>
TVector STDVectorToEigenVector(const std::vector<typename TVector::Scalar>& vec)
{
  TVector eigenVector(vec.size());
  for(unsigned int i = 0; i < static_cast<unsigned int>(vec.size()); ++i)
  {
    eigenVector[i] = vec[i];
  }
  // Could alternatively use an Eigen Map: http://eigen.tuxfamily.org/dox-devel/TutorialMapClass.html
  return eigenVector;
}

template <typename TVector>
std::vector<typename TVector::Scalar> EigenVectorToSTDVector(const TVector& vec)
{
  std::vector<float> stdvector(vec.size());
  for(unsigned int i = 0; i < static_cast<unsigned int>(vec.size()); ++i)
  {
    stdvector[i] = vec[i];
  }
  // Could equivalently do:
  // std::vector<float> stdvector(vec.data(), vec.data()+vec.size());
  return stdvector;
}

template <typename TVectorOfVectors>
void OutputVectors(const TVectorOfVectors& vectors)
{
  for(unsigned int i = 0; i < vectors.size(); ++i)
  {
    std::cout << vectors[i] << std::endl;
  }
}

template <typename TMatrix>
TMatrix TruncateRows(const TMatrix& m, const unsigned int numberOfRowsToKeep)
{
  TMatrix truncated = TMatrix::Zero(numberOfRowsToKeep, m.cols());

  for(int r = 0; r < truncated.rows(); ++r)
  {
    truncated.row(r) = m.row(r);
  }

  return truncated;
}

template <typename TMatrix>
TMatrix TruncateColumns(const TMatrix& m, const unsigned int numberOfColumnsToKeep)
{
  TMatrix truncated = TMatrix::Zero(m.rows(), numberOfColumnsToKeep);

  for(int c = 0; c < truncated.cols(); ++c)
  {
    truncated.col(c) = m.col(c);
  }

  return truncated;
}

template <typename TMatrix>
void OutputMatrixSize(const TMatrix& m)
{
  std::cout << m.rows() << "x" << m.cols() << std::endl;
}

template <typename TVectorOfVectors>
typename TVectorOfVectors::value_type ComputeMeanVector(const TVectorOfVectors& vectors)
{
  if(vectors.size() == 0)
  {
    throw std::runtime_error("Can't compute the mean of a list of vectors of length zero!");
  }
  // Compute mean vector
  typename TVectorOfVectors::value_type meanVector = TVectorOfVectors::value_type::Zero(vectors[0].size());
  for(unsigned int i = 0; i < vectors.size(); ++i)
  {
    meanVector += vectors[i];
  }

  meanVector /= vectors.size();

  return meanVector;
}

template <typename TVectorOfVectors>
typename TVectorOfVectors::value_type ComputeMinVector(const TVectorOfVectors& vectors)
{
  if(vectors.size() == 0)
  {
    throw std::runtime_error("Can't compute the mean of a list of vectors of length zero!");
  }

  typename TVectorOfVectors::value_type minVector = TVectorOfVectors::value_type::Zero(vectors[0].size());
  for(int dim = 0; dim < vectors[0].size(); ++dim) // loop through each dimension
  {
    std::vector<typename TVectorOfVectors::value_type::Scalar> values(vectors[0].size());
    for(unsigned int i = 0; i < vectors.size(); ++i)
    {
      values[i] = vectors[i](dim);
    }
    minVector[dim] = *(std::min_element(values.begin(), values.end()));
  }

  return minVector;
}

template <typename TVectorOfVectors>
typename TVectorOfVectors::value_type ComputeMaxVector(const TVectorOfVectors& vectors)
{
  if(vectors.size() == 0)
  {
    throw std::runtime_error("Can't compute the mean of a list of vectors of length zero!");
  }

  typename TVectorOfVectors::value_type maxVector = TVectorOfVectors::value_type::Zero(vectors[0].size());
  for(int dim = 0; dim < vectors[0].size(); ++dim) // loop through each dimension
  {
    std::vector<typename TVectorOfVectors::value_type::Scalar> values(vectors[0].size());
    for(unsigned int i = 0; i < vectors.size(); ++i)
    {
      values[i] = vectors[i](dim);
    }
    maxVector[dim] = *(std::max_element(values.begin(), values.end()));
  }

  return maxVector;
}

template <typename TMatrix, typename TVectorOfVectors>
TMatrix ConstructCovarianceMatrix(const TVectorOfVectors& vectors)
{
  unsigned int numberOfVectors = vectors.size();

  if(numberOfVectors == 0)
  {
    throw std::runtime_error("Can't compute the covariance matrix of a list of vectors of length zero!");
  }

  unsigned int numberOfDimensions = vectors[0].size();

  typename TVectorOfVectors::value_type meanVector = ComputeMeanVector(vectors);
  for(int i = 0; i < meanVector.size(); ++i)
  {
    if(meanVector[i] != meanVector[i]) // check for NaN
    {
      throw std::runtime_error("meanVector cannot contain NaN!");
    }
  }
  // std::cout << "meanVector: " << meanVector << std::endl;

  // Construct covariance matrix
  TMatrix covarianceMatrix = TMatrix::Zero(numberOfDimensions, numberOfDimensions);
  // std::cout << "covarianceMatrix size: " << covarianceMatrix.rows() << " x " << covarianceMatrix.cols() << std::endl;

  for(unsigned int i = 0; i < numberOfVectors; ++i)
  {
    covarianceMatrix += (vectors[i] - meanVector) * (vectors[i] - meanVector).transpose();
  }

  covarianceMatrix /= (numberOfVectors - 1); // this is the "N-1" for an unbiased estimate

  return covarianceMatrix;
}

template <typename TMatrix, typename TVectorOfVectors>
TMatrix ConstructCovarianceMatrixZeroMeanFast(const TVectorOfVectors& vectors)
{
  unsigned int numberOfVectors = vectors.size();
  unsigned int numberOfDimensions = vectors[0].size();
  TMatrix featureMatrix(numberOfDimensions, numberOfVectors);

  for(unsigned int vectorId = 0; vectorId < numberOfVectors; ++vectorId)
  {
    featureMatrix.col(0) = vectors[0];
  }
//   for(unsigned int dimension = 0; dimension < numberOfDimensions; ++dimension)
//   {
//     for(unsigned int vectorId = 0; vectorId < numberOfVectors; ++vectorId)
//     {
//       featureMatrix(dimension, vectorId) = vectors[vectorId][dimension];
//     }
//   }

  std::cout << "Done creating feature matrix." << std::endl;

  // This is the naive method
  //TMatrix covarianceMatrix = (1.0f / static_cast<typename TMatrix::Scalar>(numberOfVectors)) * featureMatrix * featureMatrix.transpose();

  // This method only computes half of the covariance matrix because it is symmetric
  TMatrix covarianceMatrix(featureMatrix.rows(), featureMatrix.rows());
  covarianceMatrix.template selfadjointView<Eigen::Upper>().rankUpdate(featureMatrix);

  // Normalize
  covarianceMatrix *= (1.0f / static_cast<typename TMatrix::Scalar>(numberOfVectors - 1));
  
  return covarianceMatrix;
}

template <typename TMatrix>
TMatrix ConstructCovarianceMatrixFromFeatureMatrix(const TMatrix& featureMatrix)
{
  // This method only computes half of the covariance matrix because it is symmetric
  TMatrix covarianceMatrix(featureMatrix.rows(), featureMatrix.rows());
  covarianceMatrix.template selfadjointView<Eigen::Upper>().rankUpdate(featureMatrix);

  float normalizationFactor = (1.0f / static_cast<typename TMatrix::Scalar>(featureMatrix.cols() - 1));
  covarianceMatrix *= normalizationFactor;
  
  // This is the naive method
  //TMatrix covarianceMatrix = normalizationFactor * featureMatrix * featureMatrix.transpose();
  
  return covarianceMatrix;
}

template <typename TMatrix, typename TVectorOfVectors>
TMatrix ConstructCovarianceMatrixZeroMean(const TVectorOfVectors& vectors)
{
  unsigned int numberOfVectors = vectors.size();

  if(numberOfVectors == 0)
  {
    throw std::runtime_error("Can't compute the covariance matrix of a list of vectors of length zero!");
  }

  unsigned int numberOfDimensions = vectors[0].size();

  // std::cout << "meanVector: " << meanVector << std::endl;

  // Construct covariance matrix
  TMatrix covarianceMatrix = TMatrix::Zero(numberOfDimensions, numberOfDimensions);
  // std::cout << "covarianceMatrix size: " << covarianceMatrix.rows() << " x " << covarianceMatrix.cols() << std::endl;

  for(unsigned int i = 0; i < numberOfVectors; ++i)
  {
    //printf("Processing %d of %d\n", i, vectors.size());
    covarianceMatrix += vectors[i] * vectors[i].transpose();
  }

  covarianceMatrix /= (numberOfVectors - 1); // this is the "N-1" for an unbiased estimate

  return covarianceMatrix;
}

template <typename TMatrix, typename TVector>
std::vector<TVector, Eigen::aligned_allocator<TVector> >
DimensionalityReduction(const std::vector<TVector, Eigen::aligned_allocator<TVector> >& vectors,
                        const TMatrix& covarianceMatrix,
                        const unsigned int numberOfDimensions)
{
  typedef Eigen::JacobiSVD<TMatrix> SVDType;
  SVDType svd(covarianceMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // Only keep the first N singular vectors of U
  TMatrix truncatedU = TruncateColumns(svd.matrixU(), numberOfDimensions);

  std::vector<TVector, Eigen::aligned_allocator<TVector> > projected;
  for(unsigned int i = 0; i < vectors.size(); ++i)
  {
    projected.push_back(truncatedU.transpose() * vectors[i]);
  }

  return projected;
}

template <typename TMatrix, typename TVectorOfVectors>
TVectorOfVectors DimensionalityReduction(const TVectorOfVectors& vectors,
                                         const TMatrix& covarianceMatrix,
                                         const float singularValueWeightToKeep)
{
  // Compute the SVD
  typedef Eigen::JacobiSVD<TMatrix> SVDType;
  SVDType svd(covarianceMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

  //std::cout << "There are " << svd.singularValues().size() << " singular values." << std::endl;

  unsigned int numberOfDimensions = ComputeNumberOfSingularValuesToKeep(svd.singularValues(), singularValueWeightToKeep);

  // Only keep the first N singular vectors of U
  TMatrix truncatedU = TruncateColumns(svd.matrixU(), numberOfDimensions);

  TVectorOfVectors projected;
  for(unsigned int i = 0; i < vectors.size(); ++i)
  {
    projected.push_back(truncatedU.transpose() * vectors[i]);
  }

  return projected;
}

template <typename TVectorOfVectors>
TVectorOfVectors DimensionalityReduction(const TVectorOfVectors& vectors,
                                         const unsigned int numberOfDimensions)
{
  Eigen::MatrixXf covarianceMatrix = ConstructCovarianceMatrix<Eigen::MatrixXf, TVectorOfVectors>(vectors);
  std::cout << "covarianceMatrix: " << covarianceMatrix << std::endl;

  return DimensionalityReduction(vectors, covarianceMatrix, numberOfDimensions);
}

template <typename TVectorOfVectors>
void Standardize(TVectorOfVectors& vectors, typename TVectorOfVectors::value_type& meanVector,
                 typename TVectorOfVectors::value_type& standardDeviationVector)
{
  // Subtract the mean and divide by the standard devation of each set of corresponding elements.

  unsigned int numberOfVectors = vectors.size();

  if(numberOfVectors == 0)
  {
    throw std::runtime_error("Can't Standardize() a list of zero vectors!");
  }

  unsigned int numberOfDimensions = vectors[0].size();

  // Each component of the result is the mean of the corresponding collection of elements. That is,
  // the 0th component of 'meanVector' is the mean of all of the 0th components in 'vectors'.
  meanVector = ComputeMeanVector(vectors);

  for(int i = 0; i < meanVector.size(); ++i)
  {
    if(meanVector[i] != meanVector[i]) // check for NaN
      {
        throw std::runtime_error("Standardize: meanVector cannot contain NaN!");
      }
  }
  // Variance = 1/NumPixels * sum_i (x_i - u)^2
  standardDeviationVector.resize(numberOfDimensions);

  // Loop over each element
  for(int element = 0; element < static_cast<int>(numberOfDimensions); ++element)
  {
    float sumOfDifferenceFromMean = 0.0f;
    for(unsigned int i = 0; i < vectors.size(); ++i)
    {
      sumOfDifferenceFromMean += pow(vectors[i][element] - meanVector[element], 2.0f);
    }

    if(sumOfDifferenceFromMean == 0)
    {
      std::stringstream ss;
      ss << "Standardize: sumOfDifferenceFromMean cannot be zero! Channel "
         << element << " has mean " << meanVector[element] << " and vector0 has value " << vectors[0][element];
      throw std::runtime_error(ss.str());
    }

    float standardDeviation = sqrt(sumOfDifferenceFromMean / static_cast<float>(numberOfVectors - 1)); // this is the "N-1" normalization to get an unbiased estimate
    if(standardDeviation == 0)
      {
        std::stringstream ss;
        ss << "Standardize: standardDeviation cannot be zero! Channel " << element << " is 0!";
        throw std::runtime_error(ss.str());
      }
    standardDeviationVector[element] = standardDeviation;
  }

  // Actually subtract the mean and divide by the standard deviation
  for(int element = 0; element < static_cast<int>(numberOfDimensions); ++element)
  {
    for(size_t i = 0; i < vectors.size(); ++i)
    {
      vectors[i][element] -= meanVector[element];
      vectors[i][element] /= standardDeviationVector[element];
    }
  }
}

template <typename TVectorOfVectors>
void Standardize(TVectorOfVectors& vectors)
{
  typename TVectorOfVectors::value_type meanVector;
  typename TVectorOfVectors::value_type standardDeviationVector;
  Standardize(vectors, meanVector, standardDeviationVector);
}

template <typename TMatrix, typename TVector>
TVector DimensionalityReduction(const TVector& v, const TMatrix& U,
                                const TVector& singularValues,
                                const float singularValueWeightToKeep)
{
  unsigned int numberOfDimensions = ComputeNumberOfSingularValuesToKeep(singularValues, singularValueWeightToKeep);
  return DimensionalityReduction(v, U, numberOfDimensions);
}

template <typename TMatrix, typename TVector>
TVector DimensionalityReduction(const TVector& v, const TMatrix& U, const unsigned int numberOfDimensions)
{
  // Only keep the first N singular vectors of U
  TMatrix truncatedU = TruncateColumns(U, numberOfDimensions);

  return truncatedU.transpose() * v;
}

template <typename TVector>
unsigned int ComputeNumberOfSingularValuesToKeep(const TVector& singularValues, const float singularValueWeightToKeep)
{
  float singularValueSum = 0.0f;
  //SVDType::SingularValuesType singularValues = svd.singularValues();
  std::cout << "SingularValues: ";
  for(int i = 0; i < singularValues.size(); ++i)
  {
    singularValueSum += singularValues[i];
    std::cout << singularValues[i] << " ";
  }

  std::cout << std::endl;
  // Determine how many vectors we need to keep to keep the desired amount of "eigen weight"

  float normalizedSingularVectorSum = 0.0f;
  unsigned int numberOfDimensions = 0;
  for(int i = 0; i < singularValues.size(); ++i)
  {
    numberOfDimensions++;
    normalizedSingularVectorSum += singularValues[i]/singularValueSum;
    if(normalizedSingularVectorSum > singularValueWeightToKeep)
    {
      break;
    }
  }

  return numberOfDimensions;
}

template <typename TMatrix>
TMatrix PseudoInverse(const TMatrix &a)
{
  double epsilon = std::numeric_limits<typename TMatrix::Scalar>::epsilon();

  if(a.rows()<a.cols())
  {
    TMatrix aT = a.transpose();
    return PseudoInverse(aT).transpose(); // For some reason the compiler doesn't like PseudoInverse(a.transpose()).transpose()
  }
    Eigen::JacobiSVD<TMatrix> svd =
         a.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

  typename TMatrix::Scalar tolerance = epsilon * std::max(a.cols(), a.rows()) *
                                      svd.singularValues().array().abs().maxCoeff();

  return svd.matrixV() * TMatrix( (svd.singularValues().array().abs() >
         tolerance).select(svd.singularValues().
         array().inverse(), 0) ).asDiagonal() * svd.matrixU().adjoint();
}

template <typename TVector>
void OutputHorizontal(const std::string& name, const TVector& v)
{
  std::cout << name << ": ";
  OutputHorizontal(v);
}

template <typename TVector>
void OutputHorizontal(const TVector& v)
{
  for(int i = 0; i < v.size(); ++i)
  {
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
}

template <typename TVector>
TVector ScaleVector(const TVector& v, const typename TVector::Scalar& lower,
                    const typename TVector::Scalar& upper)
{
  std::vector<typename TVector::Scalar> values(v.size());
  for(int i = 0; i < v.size(); ++i)
  {
    values[i] = v[i];
  }
  typename TVector::Scalar minValue = *(std::min_element(values.begin(), values.end()));
  typename TVector::Scalar maxValue = *(std::max_element(values.begin(), values.end()));

  typename TVector::Scalar outputRange = upper - lower;
  typename TVector::Scalar valueRange = maxValue - minValue;

  TVector outputVector(v.size());

  typename TVector::Scalar scaleFactor = outputRange/valueRange;
  for(int i = 0; i < v.size(); ++i)
  {
    outputVector[i] = (v[i] - minValue) * scaleFactor;
  }

  return outputVector;
}

template <typename TVector>
TVector RandomUnitVector(const unsigned int dim)
{
  TVector randomUnitVector(dim);
  for(int i = 0; i < randomUnitVector.size(); ++i)
  {
    randomUnitVector[i] = drand48() - 0.5f;
  }

  randomUnitVector.normalize();

  return randomUnitVector;
}

template <typename TPoint>
void GetBoundingBox(const std::vector<TPoint, Eigen::aligned_allocator<TPoint> >& data, TPoint& minCorner, TPoint& maxCorner)
{
  assert(data.size() > 0);

  minCorner.resize(data[0].size());
  maxCorner.resize(data[0].size());

  for(int coordinate = 0; coordinate < data[0].size(); ++coordinate)
  {
    minCorner[coordinate] = std::numeric_limits<typename TPoint::Scalar>::max();
    maxCorner[coordinate] = std::numeric_limits<typename TPoint::Scalar>::min();

    for(unsigned int pointId = 0; pointId < data.size(); ++pointId)
    {
      if(data[pointId][coordinate] > maxCorner[coordinate])
      {
        maxCorner[coordinate] = data[pointId][coordinate];
      }

      if(data[pointId][coordinate] < minCorner[coordinate])
      {
        minCorner[coordinate] = data[pointId][coordinate];
      }
    }
  }
}

} // namespace EigenHelpers

#endif
